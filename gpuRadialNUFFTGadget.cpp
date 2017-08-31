#include "gpuRadialNUFFTGadget.h"

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

//---------------------
// NFFT^H reconstruction using GPU
// implemented by Jia Sen on 2016/01/12
// Bug: the reconstructed image matrix is square
//       not reduced FOV reconstruction
//---------------------


namespace Gadgetron{
  
  gpuRadialNUFFTGadget::gpuRadialNUFFTGadget()
  : slices_(-1)
  , samples_per_profile_(-1)
  , num_coils_(0)
  , profiles_(0)
  , image_counter_(0)
  , buffer_(0)
  , slice_positions(0)
  , profiles_per_frame_(0)
  , frames_per_reconstruction_(0)
  , kernel_width_(5)
  , oversampling_factor_(2)
  , traj_mode(Trajectory2D::Uniform)
  {
  }
  
  gpuRadialNUFFTGadget::~gpuRadialNUFFTGadget()
  {
    if(buffer_) delete buffer_;
  }
  
  int gpuRadialNUFFTGadget::process_config(ACE_Message_Block* mb)
  {
    //GDEBUG("gpuRadialNUFFTGadget::process_config\n");
    
    // Setup and validate GPU cuda device configuration
    //
    
    int number_of_devices;
    if (cudaGetDeviceCount(&number_of_devices)!= cudaSuccess) {
      GDEBUG( "Error: unable to query number of CUDA devices.\n" );
      return GADGET_FAIL;
    }
    
    if (number_of_devices == 0) {
      GDEBUG( "Error: No available CUDA devices.\n" );
      return GADGET_FAIL;
    }
    
    int device_number_ = 0;
    if (device_number_ >= number_of_devices) {
      GDEBUG("Adjusting device number from %d to %d\n", device_number_,  (device_number_%number_of_devices));
      device_number_ = (device_number_%number_of_devices);
    }
    
    if (cudaSetDevice(device_number_)!= cudaSuccess) {
      GDEBUG( "Error: unable to set CUDA device.\n" );
      return GADGET_FAIL;
    }
    
    cudaDeviceProp deviceProp;
    if( cudaGetDeviceProperties( &deviceProp, device_number_ ) != cudaSuccess) {
      GDEBUG( "Error: unable to query device properties.\n" );
      return GADGET_FAIL;
    }
    
    unsigned int warp_size = deviceProp.warpSize;
    
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
    
    
    //---------------------------------------------//
    // Encoding & Reconstruction Matrix Size
    //---------------------------------------------//
    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);
    
    image_dimensions_original_.push_back(r_space.matrixSize.x);
    image_dimensions_original_.push_back(r_space.matrixSize.y);
    image_dimensions_original_.push_back(r_space.matrixSize.z);
    
    
    GDEBUG("encoding matrix_size_x : %d, y: %d, z: %d \n", 
	   dimensions_[0], dimensions_[1], dimensions_[2]);
    
    GDEBUG("original recon matrix_size_x : %d, y: %d, z: %d \n", 
	   image_dimensions_original_[0], image_dimensions_original_[1], image_dimensions_original_[2]);
    
    //---------------------------------------------//
    // Matrix sizes for gpu NUFFT plan (as a multiple of the GPU's warp size)
    // warp size requirement is due to the cuNFFTplan 
    //---------------------------------------------//
    image_dimensions_.push_back(((r_space.matrixSize.x+warp_size-1)/warp_size)*warp_size);
    image_dimensions_.push_back(((r_space.matrixSize.y+warp_size-1)/warp_size)*warp_size);
    
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(r_space.matrixSize.x*reconstruction_os_factor_x))+warp_size-1)/warp_size)*warp_size);  
    image_dimensions_recon_.push_back(((static_cast<unsigned int>(std::ceil(r_space.matrixSize.x*reconstruction_os_factor_y))+warp_size-1)/warp_size)*warp_size);
    
    image_dimensions_recon_os_ = uint64d2
    (((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[0]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size,
     ((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[1]*oversampling_factor_))+warp_size-1)/warp_size)*warp_size);
    
    // In case the warp_size constraint kicked in
    oversampling_factor_ = float(image_dimensions_recon_os_[0])/float(image_dimensions_recon_[0]); 
    
    GDEBUG("warpped matrix_size_x : %d, recon: %d, recon_os: %d\n", 
	   image_dimensions_[0], image_dimensions_recon_[0], image_dimensions_recon_os_[0]);
    
    GDEBUG("warpped matrix_size_y : %d, recon: %d, recon_os: %d\n", 
	   image_dimensions_[1], image_dimensions_recon_[1], image_dimensions_recon_os_[1]);
    
    
    fov_.push_back(r_space.fieldOfView_mm.x);
    fov_.push_back(r_space.fieldOfView_mm.y);
    fov_.push_back(r_space.fieldOfView_mm.z);
    
    slices_   = e_limits.slice ? e_limits.slice->maximum + 1 : 1;
    profiles_ = e_limits.kspace_encoding_step_1 ? e_limits.kspace_encoding_step_1->maximum + 1 : 1;
    samples_per_profile_ = e_space.matrixSize.x;
    
    //-----------------------------//
    // Calculate Trajectory and DCF
    //-----------------------------//
    traj.initializeTrajectory(Trajectory2D::HalfRange, traj_mode, profiles_, image_dimensions_original_[0], image_dimensions_original_[1]);
    traj.calculateTrajectory();
    
    this->cal_radial2D_dcf (dcf, image_dimensions_recon_[0], image_dimensions_recon_[1]);
    
    GDEBUG("encoding limits slices_: %d, profiles_: %d \n", slices_, profiles_);
    
    
    return GADGET_OK;
  }
  
  int gpuRadialNUFFTGadget::
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
    
    //---------------------------------------------------------//
    // Create buffer storing sampled data
    // dim1: Readout, dim2: Profile, dim3: Channel, dim4: Slice
    //---------------------------------------------------------//
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
      
      
      //---------------------------------------------------------------//
      // calculate kspace trajectory for each samples and all profiles 
      //---------------------------------------------------------------//
      boost::shared_ptr< hoNDArray<_reald2> > all_traj ( new hoNDArray<_reald2>(samples_per_profile_,profiles_) );
      arma::fvec fepoints = arma::linspace<arma::fvec>(-0.5, 0.5-1/float(samples_per_profile_), samples_per_profile_);     
      for (size_t p = 0; p < profiles_; p ++){
	for (size_t n = 0; n < samples_per_profile_; n ++){
	  size_t offset  = p*samples_per_profile_ + n;
	  _reald2 temp_pos;
	  temp_pos[0] = fepoints(n)*cos(traj.getAzimuthalAngle(p));
	  temp_pos[1] = fepoints(n)*sin(traj.getAzimuthalAngle(p));
	  all_traj->get_data_ptr()[offset] = temp_pos;
	}
      } 
      
      //---------------------------------------------------------------------//
      // 2DT reconstruction
      // profiles_per_frame_ = profiles_ will reconstruct only one image. 
      //--------------------------------------------------------------------//
      profiles_per_frame_ = long(profiles_*0.1);
      frames_per_reconstruction_ = long (profiles_/profiles_per_frame_);
      size_t samples_per_frame_ = profiles_per_frame_*samples_per_profile_;
      
      std::cout<<"--------------------------------------------------------------"<<std::endl;
      std::cout<< "All profiles are accumulated, we are ready to reconstruct !"<<std::endl;
      std::cout<<"Profiles:"<<profiles_<<"  Slices:"<<slices_<< std::endl;
      std::cout<<"Samples_per_profiles:"<<samples_per_profile_<<" Profiles_per_frame:"<< profiles_per_frame_ <<"  frames_per_reconstruction:"<< frames_per_reconstruction_ <<std::endl;
      std::cout<<"--------------------------------------------------------------"<<std::endl;
      
      //-------------------------------//
      // Initializing GPU NUFFT plan
      //-------------------------------//
      GPUTimer *timer;
      // Configuration image dimensions_
      uint64d2 matrix_size = uint64d2(image_dimensions_recon_[0],image_dimensions_recon_[1]);
      uint64d2 matrix_size_os = uint64d2(image_dimensions_recon_os_[0],image_dimensions_recon_os_[1]);
      
      _real kernel_width = 5.5;
      _real alpha = (_real)matrix_size_os.vec[0]/(_real)matrix_size.vec[0];
      
      // Initialize plan
      timer = new GPUTimer("Initializing plan");
      cuNFFT_plan<_real,2> plan( matrix_size, matrix_size_os, kernel_width );
      delete timer;
      
      for(unsigned int islice = 0; islice<slices_; islice++){
	
	// prepare data, traj, dcw
	std::vector<size_t> dim_array;
	dim_array.push_back(samples_per_frame_);
	dim_array.push_back(frames_per_reconstruction_);
	boost::shared_ptr< hoNDArray<_complext> > host_samples ( new hoNDArray<_complext>(&dim_array));
	boost::shared_ptr< hoNDArray<_reald2> >   host_traj    ( new hoNDArray<_reald2>(&dim_array) );
	boost::shared_ptr< hoNDArray<_real> >     host_dcw     ( new hoNDArray<_real>(&dim_array) );
	
	Gadgetron::hoNDArray< std::complex<float> > coilIm;	coilIm.create(image_dimensions_recon_[0],image_dimensions_recon_[1],frames_per_reconstruction_,num_coils_);
	std::complex<float>* d_coilIm = coilIm.get_data_ptr();
        timer = new GPUTimer("GPU gridding and FFT");
	for (unsigned int iter_coil = 0; iter_coil<num_coils_; iter_coil++){
	  
	  for(unsigned int iter_frame = 0; iter_frame<frames_per_reconstruction_; iter_frame++){
	    
	    size_t profile_offset = iter_frame*profiles_per_frame_;
	    size_t src_offset = islice*buffer_dims_[0]*buffer_dims_[1]*buffer_dims_[2]+iter_coil*buffer_dims_[0]*buffer_dims_[1] + profile_offset*buffer_dims_[0];
	    
	    // samples
	    memcpy(host_samples->get_data_ptr()+iter_frame*samples_per_frame_, b+src_offset, sizeof(_complext)*samples_per_frame_);
	    
	    // traj
	    memcpy(host_traj->get_data_ptr()+iter_frame*samples_per_frame_, all_traj->get_data_ptr() + profile_offset*samples_per_profile_,sizeof(_reald2)*samples_per_frame_);
	    
	    // dcw
	    for (unsigned int iter_sample = 0; iter_sample<samples_per_frame_; iter_sample++){
	      host_dcw->get_data_ptr()[iter_frame*samples_per_frame_+iter_sample] = dcf[iter_frame*samples_per_frame_+iter_sample];
	    }
	  }
	  
	  // Setup resulting image array
	  vector<size_t> image_dims = to_std_vector(matrix_size); 
	  image_dims.push_back(frames_per_reconstruction_);
	  cuNDArray<_complext> image(&image_dims);
	  clear(&image);
	  
	  
	  // Upload arrays to device -- The total reconstruction maybe splitted along frame dimension
	  cuNDArray<_complext> _samples(host_samples.get());
	  cuNDArray<_reald2> _trajectory(host_traj.get());
	  cuNDArray<_real> dcw(host_dcw.get());
	  
	  // split the total reconstruction to several sub-reconstruction along frame dimension
	  
	  int frames_per_reconstruction = frames_per_reconstruction_;
	  std::vector<size_t> dims_recon;
	  dims_recon.push_back(host_samples->get_size(0));
	  dims_recon.push_back(frames_per_reconstruction);
	  
	  for( unsigned int iteration = 0; iteration < frames_per_reconstruction_/frames_per_reconstruction; iteration++ ) {
	    
	    // Set samples/trajectory for sub-frames
	    cuNDArray<_complext> samples( dims_recon, _samples.get_data_ptr()+iteration*dims_recon[0]*dims_recon[1] );
	    cuNDArray<_reald2> trajectory( dims_recon, _trajectory.get_data_ptr()+iteration*dims_recon[0]*dims_recon[1] );
	    
	    // Preprocess
	    plan.preprocess( &trajectory, plan_type::NFFT_PREP_NC2C );
	    
	    std::vector<size_t> image_dims = to_std_vector(matrix_size); 
	    image_dims.push_back(frames_per_reconstruction);
	    cuNDArray<_complext> tmp_image(&image_dims, image.get_data_ptr()+iteration*prod(matrix_size)*frames_per_reconstruction);
	    
	    // Gridder
	    plan.compute( &samples, &tmp_image, &dcw, plan_type::NFFT_BACKWARDS_NC2C );
	    
	  }
	  
	  // Output result: from device to host
	  boost::shared_ptr< hoNDArray<_complext> > host_image = image.to_host();
	  
	  size_t im_offset = image_dimensions_recon_[0]*image_dimensions_recon_[1]*frames_per_reconstruction_; memcpy(d_coilIm+iter_coil*im_offset,host_image->get_data_ptr(),sizeof(_complext)*im_offset);
	} // end coil-by-coil gridding
	delete timer;
	
	
	//-------------------------------------------//
	// csm based coil-combination
	//-------------------------------------------//
	for(unsigned int iframe = 0; iframe<frames_per_reconstruction_; iframe++){
	  
	  Gadgetron::hoNDArray< std::complex<float> > coilIm_frame;
	  coilIm_frame.create(image_dimensions_recon_[0],image_dimensions_recon_[1],num_coils_);
	  std::complex<float>* d_coilIm_frame = coilIm_frame.get_data_ptr();
	  
	  size_t image_pixel_num = image_dimensions_recon_[0]*image_dimensions_recon_[1];
	  size_t frame_pixel_num = image_pixel_num*frames_per_reconstruction_;
	  for(unsigned int icoil = 0; icoil<num_coils_; icoil++){
	    memcpy(d_coilIm_frame+icoil*image_pixel_num, d_coilIm+icoil*frame_pixel_num+iframe*image_pixel_num, sizeof(std::complex<float>)*image_pixel_num);
	  }
	  
	  /* Calculate coil sensitivity map and coil map weighted coil combination */
	  Gadgetron::gtPlus::gtPlusISMRMRDReconUtilComplex<std::complex<float> > gtPlus_util_complex_;
	  Gadgetron::ISMRMRDCOILMAPALGO algo = Gadgetron::ISMRMRD_SOUHEIL;
	  unsigned long long csm_kSize_ = 7;
	  unsigned long long csm_powermethod_num_ = 3;
	  Gadgetron::hoNDArray< std::complex<float> > coilMap;  
	  gtPlus_util_complex_.coilMap2DNIH(coilIm_frame, coilMap, algo, csm_kSize_,csm_powermethod_num_, false);
// 	  write_nd_array<std::complex<float> >(&coilMap,"coilMap_full.cplx");
// 	  write_nd_array<std::complex<float> >(&coilIm_frame,"coilIm_frame.cplx");
	  
	  Gadgetron::hoNDArray<std::complex< float> > acc_im;
	  gtPlus_util_complex_.coilCombine(coilIm_frame, coilMap, acc_im);
	  std::complex<float>* d_accim = acc_im.get_data_ptr();
	  
	  // Create a new message with an hoNDArray for the combined image
	  GadgetContainerMessage< hoNDArray<std::complex<float> > >* m8 = 
	  new GadgetContainerMessage< hoNDArray<std::complex<float> > >();
	  m8->getObjectPtr()->create(&image_dimensions_recon_);
	  std::complex<float>* d_sosim = m8->getObjectPtr()->get_data_ptr();
	  
	  
	  // 2D image
	  int nz = 1;
	  int ny = image_dimensions_recon_[1];
	  int nx = image_dimensions_recon_[0];
	  
	  for (size_t z = 0; z < nz; z++) {
	    for (size_t y = 0; y < ny; y++) {
	      for (size_t x = 0; x < nx; x++) {
		size_t offset = z*ny*nx+y*nx+x;
		d_sosim[offset] = d_accim[offset]; 
	      }
	    }
	  }
	  
	  //---------------------//
	  // sending out image
	  //---------------------//
	  // Modify header to match the size and change the type to real
	  // Prepare the image header
	  GadgetContainerMessage<ISMRMRD::ImageHeader> *m7 = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
	  ISMRMRD::AcquisitionHeader *base_head = m1->getObjectPtr();
	  
	  // Initialize header to all zeroes (there is a few fields we do not set yet)
	  ISMRMRD::ImageHeader tmp;
	  *(m7->getObjectPtr()) = tmp;
	  
	  m7->getObjectPtr()->version = base_head->version;
	  m7->getObjectPtr()->flags   = base_head->flags;
	  m7->getObjectPtr()->measurement_uid = base_head->measurement_uid;
	  
	  m7->getObjectPtr()->matrix_size[0] = image_dimensions_recon_[0];
	  m7->getObjectPtr()->matrix_size[1] = image_dimensions_recon_[1];
	  m7->getObjectPtr()->matrix_size[2] = 1;
	  
	  m7->getObjectPtr()->field_of_view[0] = fov_[0];
	  m7->getObjectPtr()->field_of_view[1] = fov_[1];
	  m7->getObjectPtr()->field_of_view[2] = fov_[2];
	  
	  m7->getObjectPtr()->channels = 1;
	  m7->getObjectPtr()->slice    = islice;
	  m7->getObjectPtr()->set      = base_head->idx.set;
	  m7->getObjectPtr()->phase    = iframe;
	  
	  m7->getObjectPtr()->acquisition_time_stamp = base_head->acquisition_time_stamp;
	  memcpy(m7->getObjectPtr()->physiology_time_stamp, base_head->physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);
	  
	  m7->getObjectPtr()->position[0] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+0];
	  m7->getObjectPtr()->position[1] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+1];
	  m7->getObjectPtr()->position[2] = slice_positions[islice*(ISMRMRD::ISMRMRD_POSITION_LENGTH)+2];
	  memcpy(m7->getObjectPtr()->read_dir,  base_head->read_dir, sizeof(float)*3);
	  memcpy(m7->getObjectPtr()->phase_dir, base_head->phase_dir, sizeof(float)*3);
	  memcpy(m7->getObjectPtr()->slice_dir, base_head->slice_dir, sizeof(float)*3);
	  memcpy(m7->getObjectPtr()->patient_table_position, base_head->patient_table_position, sizeof(float)*3);
	  
	  m7->getObjectPtr()->data_type   = ISMRMRD::ISMRMRD_CXFLOAT;
	  m7->getObjectPtr()->image_type  = ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
	  m7->getObjectPtr()->image_index = islice*frames_per_reconstruction_+iframe+1; 
	  m7->getObjectPtr()->image_series_index = 1;
	  
	  memcpy(m7->getObjectPtr()->user_int,   m1->getObjectPtr()->user_int, sizeof(int32_t)*8);
	  memcpy(m7->getObjectPtr()->user_float, m1->getObjectPtr()->user_float, sizeof(float)*8);
	  
	  // Now add the new array to the outgoing message
	  m7->cont(m8);
	  // 	this->next()->putq(m5);
	  if (this->next()->putq(m7) < 0){
	    GDEBUG("Failed to put image into message queue \n");
	    m7->release();
	    return GADGET_FAIL;
	  }
	  
	} // end of inner iframe loop for coil combination
	
      } // End of outer slice loop for slice-by-slice reconstruction
      
    } // End of is_allprofiles_loaded for whole reconstruction work
    
    m1->release();
    return GADGET_OK;
  }
  
  bool gpuRadialNUFFTGadget::cal_radial2D_dcf ( std::vector<float>& dcf, double fov1, double fov2)
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
  
  
  GADGET_FACTORY_DECLARE(gpuRadialNUFFTGadget)
}

