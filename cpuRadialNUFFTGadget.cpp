#include "cpuRadialNUFFTGadget.h"

#include "hoNDArray_elemwise.h"
#include "hoNDArray_fileio.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <complex>

#include <math.h>
#include <stdlib.h>


#include "gtPlusISMRMRDReconUtil.h"

#ifdef USE_OMP
#include <omp.h>
#endif // USE_OMP


namespace Gadgetron{
  
  cpuRadialNUFFTGadget::cpuRadialNUFFTGadget()
  : slices_(-1)
  , samples_per_profile_(-1)
  , num_coils_(0)
  , profiles_(0)
  , image_counter_(0)
  , buffer_(0)
  , slice_positions(0)
  , profiles_per_frame_(0)
  , frames_per_reconstruction_(0)
  , traj_mode(Trajectory2D::Uniform)
  {
  }
  
  cpuRadialNUFFTGadget::~cpuRadialNUFFTGadget()
  {
    if(buffer_) delete buffer_;
  }
  
  int cpuRadialNUFFTGadget::process_config(ACE_Message_Block* mb)
  {
    //GDEBUG("cpuRadialNUFFTGadget::process_config\n");
    
    
    // Get the Ismrmrd header
    //
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);
    
    if (h.encoding.size() != 1) {
      GDEBUG("This Gadget only supports one encoding space\n");
      return GADGET_FAIL;
    }
    
    // Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace  e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace  r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc = *h.encoding[0].trajectoryDescription;
    
    for (std::vector<ISMRMRD::UserParameterLong>::iterator i (traj_desc.userParameterLong.begin()); i != traj_desc.userParameterLong.end(); ++i) {
      if (i->name == "RadialTrajMode") {
	if (i->value == 5)
	  traj_mode = Trajectory2D::GoldenAngle;
	else
	  traj_mode = Trajectory2D::Uniform;
      }else if (i->name == "FibonacciN"){
	traj.setlFibonacciN(i->value);
      }else if (i->name == "myRadialViews"){
      }
    }
    
    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);
  
    image_dimensions_.push_back(r_space.matrixSize.x);
    image_dimensions_.push_back(r_space.matrixSize.y);
    image_dimensions_.push_back(r_space.matrixSize.z);
    
    
    GDEBUG("encoding matrix_size_x : %d, y: %d, z: %d \n", 
	   dimensions_[0], dimensions_[1], dimensions_[2]);
    
    GDEBUG("recon matrix_size_x : %d, y: %d, z: %d \n", 
	   image_dimensions_[0], image_dimensions_[1], image_dimensions_[2]);
    
    fov_.push_back(r_space.fieldOfView_mm.x);
    fov_.push_back(r_space.fieldOfView_mm.y);
    fov_.push_back(r_space.fieldOfView_mm.z);
    
    slices_   = e_limits.slice ? e_limits.slice->maximum + 1 : 1;
    profiles_ = e_limits.kspace_encoding_step_1 ? e_limits.kspace_encoding_step_1->maximum + 1 : 1;
    samples_per_profile_ = e_space.matrixSize.x;

    traj.initializeTrajectory(Trajectory2D::HalfRange, traj_mode, profiles_, image_dimensions_[0], image_dimensions_[1]);
    traj.calculateTrajectory();
    
    this->cal_radial2D_dcf (dcf, image_dimensions_[0], image_dimensions_[1]);
      
    GDEBUG("encoding limits slices_: %d, profiles_: %d \n", slices_, profiles_);
    
    
    return GADGET_OK;
  }
  
  int cpuRadialNUFFTGadget::
  process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2)
  {
    
    // Noise should have been consumed by the noise adjust (if in the gadget chain)
    bool is_noise = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
    if (is_noise) { 
      m1->release();
      return GADGET_OK;
    }
     
    bool is_phasecorr_data = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA);
    if (is_phasecorr_data) { 
      m1->release();
      return GADGET_OK;
    }
    
    unsigned int profile = m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice = m1->getObjectPtr()->idx.slice;
    unsigned int set = m1->getObjectPtr()->idx.set;
    unsigned int num_coils_ = m1->getObjectPtr()->active_channels;
//     std::cout<<profile<<"  "<<slice<<"  "<<set<<"  "<<std::endl;
    

    // Create buffer, copy data from m2 and store in created buffer
    if (!buffer_){
      slice_positions = std::vector<float>(slices_*(ISMRMRD::ISMRMRD_POSITION_LENGTH),0);  // positions of slice
      buffer_dims_.clear();
      buffer_dims_.push_back(samples_per_profile_);
      buffer_dims_.push_back(profiles_);
      buffer_dims_.push_back(num_coils_);
      buffer_dims_.push_back(slices_);
      if (!(buffer_ = new hoNDArray< std::complex< float > >() ) ){
	GDEBUG("Failed create buffer \n");
	return GADGET_FAIL;	  
      }
      try {buffer_->create(&buffer_dims_);}
      catch (std::runtime_error &err){
	GEXCEPTION(err,"Failed allocate data buffer array \n");
	return GADGET_FAIL;	  
      }
    }
    
    
    std::complex<float>* b = buffer_->get_data_ptr();
    std::complex<float>* d = m2->getObjectPtr()->get_data_ptr();
    size_t offset= 0;
    for (int c = 0; c < num_coils_; c++) {
      offset = slice*buffer_dims_[0]*buffer_dims_[1]*buffer_dims_[2]+c*buffer_dims_[0]*buffer_dims_[1] + profile*buffer_dims_[0];
      memcpy(b+offset, d+c*samples_per_profile_, sizeof(std::complex<float>)*samples_per_profile_);
    }
    for (int iter = 0; iter<(ISMRMRD::ISMRMRD_POSITION_LENGTH); iter++){
      slice_positions[slice*(ISMRMRD::ISMRMRD_POSITION_LENGTH) + iter] = m1->getObjectPtr()->position[iter];
    }
   
    //-----------------------------------------------------------------------------------//
    // Are we ready to reconstruct -- Use the ACQ_LAST_In_MEASUREMENT flag
    bool is_allprofiles_loaded = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
    
    if( is_allprofiles_loaded){

      //---------------------------------------------------------------------//
      // profiles_per_frame_ = profiles_ will reconstruct only one image. 
      
      profiles_per_frame_ = long(profiles_*0.1);
      frames_per_reconstruction_ = long (profiles_/profiles_per_frame_);
      
      std::cout<<"--------------------------------------------------------------"<<std::endl;
      std::cout<< "All profiles are accumulated, we are ready to reconstruct !"<<std::endl;
      std::cout<<"Profiles:"<<profiles_<<"  Slices:"<<slices_<< std::endl;
      std::cout<<"Profiles_per_frame:"<< profiles_per_frame_ <<"  frames_per_reconstruction:"<< frames_per_reconstruction_ <<std::endl;
      std::cout<<"--------------------------------------------------------------"<<std::endl;
  
      //---------------------------------------------------------------------//
      // Gridding and FFT == NUFFT for each channel
      int islice = 0;
      long iframe = 0;

      Gadgetron::GadgetronTimer gt_timer_;
      gt_timer_.start("cpuRadialNUFFTGadget::perform_gridding_frame_by_frame");
         
     
      for(islice = 0; islice<slices_; islice++){
	
 	 #pragma omp parallel for default(none) private(iframe) shared( m1, b,islice, num_coils_)
	for(iframe = 0; iframe<frames_per_reconstruction_; iframe++){
	  
	
	  nfft_plan my_plan;
	  construct_nfft_plan(my_plan,iframe);
// 	  if(!construct_nfft_plan(my_plan, iframe))
// 	  {
// 	    GDEBUG("Construct NFFT plan failed ... ");
// 	    return GADGET_FAIL;
// 	  }
// 	  
	  
	  int j = 0;
	  size_t data_offset = 0;
	  Gadgetron::hoNDArray< std::complex<float> > coilIm;
	  coilIm.create(image_dimensions_[0],image_dimensions_[1],num_coils_);
	  std::complex<float>* d_coilIm = coilIm.get_data_ptr();
	  
	  for(int icoil = 0; icoil<num_coils_; icoil++){
	    
	    /* read freal and fimag from the knots */
	    for (int iky = 0; iky<profiles_per_frame_; iky++){
	      int ky_label = iframe*profiles_per_frame_ + iky;
	      
	      for (int ikx = 0; ikx<buffer_dims_[0]; ikx++){
		data_offset = islice*buffer_dims_[0]*buffer_dims_[1]*buffer_dims_[2]+icoil*buffer_dims_[0]*buffer_dims_[1] + ky_label*buffer_dims_[0] + ikx;
		j = iky*buffer_dims_[0] + ikx;
		
		// to read the real and imag part of a complex number and densitivity compensation
		my_plan.f[j][0] = b[data_offset].real()*dcf[ky_label*buffer_dims_[0] + ikx];
		my_plan.f[j][1] = b[data_offset].imag()*dcf[ky_label*buffer_dims_[0] + ikx];    
	      }
	    }
	    
	    nfft_adjoint(&my_plan);
	    
	    size_t offset = 0;
	    //Copy the reconstructed image for all the channels
	    for (int iy = 0; iy<image_dimensions_[1]; iy++){
	      for (int ix = 0; ix<image_dimensions_[0]; ix++){
		offset = ix*image_dimensions_[1]+iy;
		data_offset = icoil*image_dimensions_[0]*image_dimensions_[1] + iy*image_dimensions_[0]+ix;
		std::complex<float> temp(my_plan.f_hat[offset][0],my_plan.f_hat[offset][1]);
		d_coilIm[data_offset] = temp;
	      }
	    }
	    
	  } // end of icoil
	  
	  /* finalize the nfft */
	  nfft_finalize(&my_plan);
	  
	  /* Calculate coil sensitivity map and coil map weighted coil combination */
	  Gadgetron::gtPlus::gtPlusISMRMRDReconUtilComplex<std::complex<float> > gtPlus_util_complex_;
	  Gadgetron::ISMRMRDCOILMAPALGO algo = Gadgetron::ISMRMRD_SOUHEIL;
	  unsigned long long csm_kSize_ = 7;
	  unsigned long long csm_powermethod_num_ = 3;
	  Gadgetron::hoNDArray< std::complex<float> > coilMap;  
	  gtPlus_util_complex_.coilMap2DNIH(coilIm, coilMap, algo, csm_kSize_,csm_powermethod_num_, false);
	  // 	if ( !gtPlus_util_complex_.coilMap2DNIH(coilIm, coilMap, algo, csm_kSize_,csm_powermethod_num_, false) )
	  // 	{
	  // 	  GDEBUG("coilMap2DNIH(...) failed ... ");
	  // 	  return GADGET_FAIL;
	  // 	}
	  
	  Gadgetron::hoNDArray<std::complex< float> > acc_im;
	  gtPlus_util_complex_.coilCombine(coilIm, coilMap, acc_im);
	  // 	if ( !gtPlus_util_complex_.coilCombine(coilIm, coilMap, acc_im) )
	  // 	{
	  // 	  GDEBUG("coilCombine(...) failed ... ");
	  // 	  return GADGET_FAIL;
	  // 	}
	  std::complex<float>* d_accim = acc_im.get_data_ptr();
	  
	  // Prepare the data arrray containing reconstructed image
	  /* Coil map based Coil combination */
	  // Create a new message with an hoNDArray for the combined image
	  GadgetContainerMessage< hoNDArray<std::complex<float> > >* m6 = 
	  new GadgetContainerMessage< hoNDArray<std::complex<float> > >();
	  m6->getObjectPtr()->create(&image_dimensions_);
	  
	  // 	try{m6->getObjectPtr()->create(&image_dimensions_);}
	  // 	catch (std::runtime_error &err){
	  // 	  GEXCEPTION(err,"CombineGadget, failed to allocate new array\n");
	  // 	  return -1;
	  // 	}
	  
	  std::complex<float>* d_sosim = m6->getObjectPtr()->get_data_ptr();
	  
	  
	  // 2D image
	  int nz = 1;
	  int ny = image_dimensions_[1];
	  int nx = image_dimensions_[0];
	  
	  for (size_t z = 0; z < nz; z++) {
	    for (size_t y = 0; y < ny; y++) {
	      for (size_t x = 0; x < nx; x++) {
		size_t offset = z*ny*nx+y*nx+x;
		d_sosim[offset] = d_accim[offset];
	      }
	    }
	  }
	  
	  // Modify header to match the size and change the type to real
	  // Prepare the image header
	  GadgetContainerMessage<ISMRMRD::ImageHeader> *m5 = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
	  ISMRMRD::AcquisitionHeader *base_head = m1->getObjectPtr();
	  
	  // Initialize header to all zeroes (there is a few fields we do not set yet)
	  ISMRMRD::ImageHeader tmp;
	  *(m5->getObjectPtr()) = tmp;
	  
	  m5->getObjectPtr()->version = base_head->version;
	  m5->getObjectPtr()->flags   = base_head->flags;
	  m5->getObjectPtr()->measurement_uid = base_head->measurement_uid;
	  
	  m5->getObjectPtr()->matrix_size[0] = image_dimensions_[0];
	  m5->getObjectPtr()->matrix_size[1] = image_dimensions_[1];
	  m5->getObjectPtr()->matrix_size[2] = image_dimensions_[2];
	  
	  m5->getObjectPtr()->field_of_view[0] = fov_[0];
	  m5->getObjectPtr()->field_of_view[1] = fov_[1];
	  m5->getObjectPtr()->field_of_view[2] = fov_[2];
	  
	  m5->getObjectPtr()->channels = 1;
	  m5->getObjectPtr()->slice    = islice;
	  m5->getObjectPtr()->set      = base_head->idx.set;
	  m5->getObjectPtr()->phase    = iframe;
	  
	  m5->getObjectPtr()->acquisition_time_stamp = base_head->acquisition_time_stamp;
	  memcpy(m5->getObjectPtr()->physiology_time_stamp, base_head->physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);
	  
	  m5->getObjectPtr()->position[0] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+0];
	  m5->getObjectPtr()->position[1] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+1];
	  m5->getObjectPtr()->position[2] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+2];
	  memcpy(m5->getObjectPtr()->read_dir,  base_head->read_dir, sizeof(float)*3);
	  memcpy(m5->getObjectPtr()->phase_dir, base_head->phase_dir, sizeof(float)*3);
	  memcpy(m5->getObjectPtr()->slice_dir, base_head->slice_dir, sizeof(float)*3);
	  memcpy(m5->getObjectPtr()->patient_table_position, base_head->patient_table_position, sizeof(float)*3);
	  
	  m5->getObjectPtr()->data_type   = ISMRMRD::ISMRMRD_CXFLOAT;
	  m5->getObjectPtr()->image_type  = ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
	  m5->getObjectPtr()->image_index = islice*frames_per_reconstruction_+iframe+1; 
	  m5->getObjectPtr()->image_series_index = 1;
	  
	  memcpy(m5->getObjectPtr()->user_int,   m1->getObjectPtr()->user_int, sizeof(int32_t)*8);
	  memcpy(m5->getObjectPtr()->user_float, m1->getObjectPtr()->user_float, sizeof(float)*8);
	  
	  // Now add the new array to the outgoing message
	  m5->cont(m6);
	  this->next()->putq(m5);
// 	  if (this->next()->putq(m5) < 0){
// 	    GDEBUG("Failed to put image into message queue \n");
// 	    m5->release();
// 	    return GADGET_FAIL;
// 	  }
// 	
	} // End of frame loop
	
      } // End of slice 
      
      gt_timer_.stop();
      
    } // End of is_allprofiles_loaded
    m1->release();
    return GADGET_OK;
  }
  
  bool cpuRadialNUFFTGadget::cal_radial2D_dcf ( std::vector<float>& dcf, double fov1, double fov2)
  {
    size_t nProfiles = profiles_; 
    size_t nSamples  = samples_per_profile_;
     
    arma::fvec fepoints = arma::linspace<arma::fvec>(-0.5, 0.5-1/float(nSamples), nSamples);  
    for (size_t p = 0; p < nProfiles; p ++){
	for (size_t iSamples = 0; iSamples < samples_per_profile_; iSamples++){
	  //dcf.push_back(abs(fepoints(iSamples))*samples_per_profile_/nProfiles);
	  dcf.push_back(abs(fepoints(iSamples))*ellipse(traj.getAzimuthalAngle(p) + M_PI/2, fov2, fov1));
	}
      }
      
    return true;
  }
  
  bool cpuRadialNUFFTGadget::construct_nfft_plan(nfft_plan& my_plan, long iframe)
  {
      size_t nProfiles = profiles_per_frame_; 
      size_t nSamples  = samples_per_profile_;
      
      size_t M = nSamples * nProfiles;
      int my_N[2],my_n[2];          
      int m = 6;
      double alpha = 2.0;
      int flags = PRE_PHI_HUT| PRE_PSI |MALLOC_X| MALLOC_F_HAT|
                      MALLOC_F| FFTW_INIT| FFT_OUT_OF_PLACE|
                      FFTW_MEASURE| FFTW_DESTROY_INPUT;
      
      my_N[0]=image_dimensions_[0]; my_n[0]=std::ceil(my_N[0]*alpha);
      my_N[1]=image_dimensions_[1]; my_n[1]=std::ceil(my_N[1]*alpha);
      
      nfft_init_guru(&my_plan, 2, my_N, M, my_n, m, flags, FFTW_MEASURE | FFTW_DESTROY_INPUT);
      
       
      arma::fvec fepoints = arma::linspace<arma::fvec>(-0.5, 0.5-1/float(nSamples), nSamples);     
      for (size_t p = 0; p < nProfiles; p ++){
	for (size_t n = 0; n < nSamples; n ++){
	  size_t offset  = p*nSamples + n;
	  int profile_label = iframe*profiles_per_frame_ + p;
	  my_plan.x[2*offset+0] = fepoints(n)*cos(traj.getAzimuthalAngle(profile_label));
	  my_plan.x[2*offset+1] = fepoints(n)*sin(traj.getAzimuthalAngle(profile_label));
	}
      }
       
      // precompute psi 
      if(my_plan.flags & PRE_PSI)
	nfft_precompute_psi(&my_plan);
      
      // precompute full psi 
      if(my_plan.flags & PRE_FULL_PSI)
	nfft_precompute_full_psi(&my_plan);
      
      return true;
  }
  
  GADGET_FACTORY_DECLARE(cpuRadialNUFFTGadget)
}

