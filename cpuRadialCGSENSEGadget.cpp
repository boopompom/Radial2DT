#include "cpuRadialCGSENSEGadget.h"

#include "hoNDArray_elemwise.h"
#include "hoNDArray_fileio.h"


#include <algorithm>
#include <vector>
#include <cmath>
#include <complex>

#include <math.h>
#include <stdlib.h>

#define NFFT_PRECISION_DOUBLE
#include "nfft3.h"

#include "gtPlusISMRMRDReconUtil.h"

#ifdef USE_OMP
#include <omp.h>
#endif // USE_OMP

#include <time.h>

namespace Gadgetron{
  
  cpuRadialCGSENSEGadget::cpuRadialCGSENSEGadget()
  : slices_(-1)
  , sets_(-1)
  , samples_per_profile_(-1)
  , num_coils_(0)
  , profiles_(0)
  , image_counter_(0)
  , buffer_(0)
  {
  }
  
  cpuRadialCGSENSEGadget::~cpuRadialCGSENSEGadget()
  {
    if(buffer_) delete buffer_;
  }
  
  int cpuRadialCGSENSEGadget::process_config(ACE_Message_Block* mb)
  {
    //GDEBUG("cpuRadialCGSENSEGadget::process_config\n");
    
    
    // Get the Ismrmrd header
    //
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);
    
    if (h.encoding.size() != 1) {
      GDEBUG("This Gadget only supports one encoding space\n");
      return GADGET_FAIL;
    }
    
    // Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;
    
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
    
    slices_ = e_limits.slice ? e_limits.slice->maximum + 1 : 1;
    sets_ = e_limits.set ? e_limits.set->maximum + 1 : 1;
    profiles_ = e_limits.kspace_encoding_step_1 ? e_limits.kspace_encoding_step_1->maximum + 1 : 1;
    
    GDEBUG("encoding limits slices_: %d, sets_: %d, profiles_: %d \n", slices_, sets_, profiles_);
    
    
    return GADGET_OK;
  }
  
  int cpuRadialCGSENSEGadget::
  process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2)
  {
    
    // Noise should have been consumed by the noise adjust (if in the gadget chain)
    bool is_noise = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
    if (is_noise) { 
      m1->release();
      return GADGET_OK;
    }
    
    unsigned int profile = m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice = m1->getObjectPtr()->idx.slice;
    unsigned int set = m1->getObjectPtr()->idx.set;
    unsigned int num_coils_ = m1->getObjectPtr()->active_channels;
    unsigned int samples_per_profile_ = m1->getObjectPtr()->number_of_samples;
    
    
    buffer_dims_.clear();
    buffer_dims_.push_back(samples_per_profile_);
    buffer_dims_.push_back(profiles_);
    buffer_dims_.push_back(num_coils_);
    
    // Create buffer, copy data from m2 and store in created buffer
    if (!buffer_){
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
      offset = c*buffer_dims_[0]*buffer_dims_[1] + profile*buffer_dims_[0];
      memcpy(b+offset, d+c*samples_per_profile_, sizeof(std::complex<float>)*samples_per_profile_);
    }
   
    //-----------------------------------------------------------------------------------//
    // Are we ready to reconstruct (downstream)?
    bool is_last_profile_in_reconstruction = (profile == profiles_-1);
    
    if( is_last_profile_in_reconstruction){
      std::cout<<" --------------------------------------------------------------------"<<std::endl;
      std::cout<< " --- All profiles are accumulated, we are ready to reconstruct ! ---"<<std::endl;
      std::cout<<" --------------------------------------------------------------------"<<std::endl;
    
      //----------------------------------------------------------------------------------//
      // calculate trajectory and densitivity compensation function
      std::vector<float> kx;
      std::vector<float> ky;
      std::vector<float> dcf;
      int mode_ = 1;
      if(!this->cal_radial2D_traj(profiles_,samples_per_profile_, kx,ky, dcf, mode_)){
	return GADGET_FAIL;
      }
      
      //----------------------------------------------------------------------------------//
      // calculate coil sensitivity map for CGSENSE
      std::vector<size_t> coil_im_dims_;
      coil_im_dims_.clear();
      coil_im_dims_.push_back(samples_per_profile_>>1);
      coil_im_dims_.push_back(samples_per_profile_>>1);
      coil_im_dims_.push_back(num_coils_);
      
      Gadgetron::hoNDArray< std::complex<float> > coilIm;
      if(!this->cal_radial2D_coilKspace_to_coilIm(buffer_, coilIm, coil_im_dims_, kx,ky, dcf)){
	return GADGET_FAIL;
      }
      std::complex<float>* d_coilIm = coilIm.get_data_ptr();
   
      Gadgetron::gtPlus::gtPlusISMRMRDReconUtilComplex<std::complex<float> > gtPlus_util_complex_;
      Gadgetron::ISMRMRDCOILMAPALGO algo = Gadgetron::ISMRMRD_SOUHEIL;
      unsigned long long csm_kSize_ = 7;
      unsigned long long csm_powermethod_num_ = 3;
      Gadgetron::hoNDArray< std::complex<float> > coilMap;  
      if ( !gtPlus_util_complex_.coilMap2DNIH(coilIm, coilMap, algo, csm_kSize_,csm_powermethod_num_, false) )
      {
	GDEBUG("coilMap2DNIH(...) failed ... ");
	return GADGET_FAIL;
      }
      std::complex<float>* d_coilMap = coilMap.get_data_ptr();
      
      //---------------------------------------------------------------------------------//
      /* CSM-based coil combination */
      Gadgetron::hoNDArray<std::complex< float> > acc_im;
      if ( !gtPlus_util_complex_.coilCombine(coilIm, coilMap, acc_im) )
      {
	GDEBUG("coilCombine(...) failed ... ");
	return GADGET_FAIL;
      }
      std::complex<float>* d_accim = acc_im.get_data_ptr();
      
      //---------------------------------------------------------------------------------//
      /* Conjugate Gradient SENSE */
      std::cout<<" --------------------------------------------------------------------"<<std::endl;
      std::cout<<" -------------------- Conjugate Gradient SENSE ! --------------------"<<std::endl;
      std::cout<<" --------------------------------------------------------------------"<<std::endl;
      
      /* Initialize a NUFFT plan */
      int profiles_per_frame_ = 100;
      
      std::vector<size_t> buffer_rec_dims_;
      buffer_rec_dims_.clear();
      buffer_rec_dims_.push_back(samples_per_profile_);
      buffer_rec_dims_.push_back(profiles_per_frame_);
      buffer_rec_dims_.push_back(num_coils_);
      
      hoNDArray< std::complex<float> >* buffer_rec_;
      if (!buffer_rec_){
	if (!(buffer_rec_ = new hoNDArray< std::complex< float > >() ) ){
	  GDEBUG("Failed create buffer \n");
	  return GADGET_FAIL;	  
	}
	try {buffer_rec_->create(&buffer_rec_dims_);}
	catch (std::runtime_error &err){
	  GEXCEPTION(err,"Failed allocate data buffer array \n");
	  return GADGET_FAIL;	  
	}
      }
      std::complex<float>* b_rec = buffer_rec_->get_data_ptr();
      size_t offset= 0;
      size_t offset1 = 0;
      for (int c = 0; c < num_coils_; c++) {
	offset = c*buffer_dims_[0]*buffer_dims_[1];
	offset1 = c*buffer_rec_dims_[0]*buffer_rec_dims_[1];
	memcpy(b_rec+offset1, b+offset, sizeof(std::complex<float>)*samples_per_profile_*profiles_per_frame_);
      }
      
      std::vector<float> kx_rec;
      std::vector<float> ky_rec;
      for (int iter=0; iter<samples_per_profile_*profiles_per_frame_;iter++){
	kx_rec.push_back(kx[iter]);
	ky_rec.push_back(ky[iter]);
      }
     
      size_t M = samples_per_profile_*profiles_per_frame_;
      int weight = 0;
      double real,imag;           /* to read the real and imag part of a complex number */
      
      int my_N[2],my_n[2];         /* to init the nfft */
      int m = 6;
      double alpha = 2.0;
      int flags = PRE_PHI_HUT| PRE_PSI |MALLOC_X| MALLOC_F_HAT|
      MALLOC_F| FFTW_INIT| FFT_OUT_OF_PLACE|
      FFTW_MEASURE| FFTW_DESTROY_INPUT;
      
      my_N[0]=coil_im_dims_[0]; my_n[0]=std::ceil(my_N[0]*alpha);
      my_N[1]=coil_im_dims_[1]; my_n[1]=std::ceil(my_N[1]*alpha);
      
      
      /* initialise my_plan */
      nfft_plan my_plan;            /* plan for the two dimensional nfft  */
      nfft_init_guru(&my_plan, 2, my_N, M, my_n, m, flags, FFTW_MEASURE | FFTW_DESTROY_INPUT);
      
      /* read nonuniform (x,y) from the trajectory */
      int j = 0;
      for (int iky = 0; iky<profiles_per_frame_; iky++){
	for (int ikx = 0; ikx<samples_per_profile_; ikx++){
	  j = iky*samples_per_profile_ + ikx;
	  my_plan.x[2*j+0] = kx_rec[j];
	  my_plan.x[2*j+1] = ky_rec[j];
	}
      }
      
      /* precompute psi */
      if(my_plan.flags & PRE_PSI)
	nfft_precompute_psi(&my_plan);
      
      /* precompute full psi */
      if(my_plan.flags & PRE_FULL_PSI)
	nfft_precompute_full_psi(&my_plan);
      
      
      /* Calculate E^H * m = S^H * F^H * m */
      Gadgetron::ho2DArray< std::complex< float > > cg_q(coil_im_dims_[0],coil_im_dims_[1]);
      cg_q.fill(0);
      Gadgetron::ho3DArray< std::complex< float > > cg_q_coil(coil_im_dims_[0],coil_im_dims_[1], num_coils_);
      cg_q_coil.fill(0);
      size_t data_offset = 0;
      for(int icoil = 0; icoil<num_coils_; icoil++){
	
	/* F^H * m*/
	for (int iky = 0; iky<profiles_per_frame_; iky++){
	  for (int ikx = 0; ikx<samples_per_profile_; ikx++){
	    data_offset = icoil*samples_per_profile_*profiles_per_frame_ + iky*samples_per_profile_ + ikx;
	    j = iky*samples_per_profile_ + ikx;
	    
	    my_plan.f[j][0] = b_rec[data_offset].real();
	    my_plan.f[j][1] = b_rec[data_offset].imag();
	    
	  }
	}
	
	nfft_adjoint(&my_plan);
	
	/* S^H * F^H * m */
	size_t offset = 0;
	for (int iy = 0; iy<coil_im_dims_[1]; iy++){
	  for (int ix = 0; ix<coil_im_dims_[0]; ix++){
	    offset = ix*coil_im_dims_[1]+iy;

	    data_offset = icoil*coil_im_dims_[0]*coil_im_dims_[1] + iy*coil_im_dims_[0]+ix;
	    
	    std::complex<float> temp(my_plan.f_hat[offset][0],my_plan.f_hat[offset][1]);
            temp = temp*std::conj(d_coilMap[data_offset]);

	    cg_q_coil[data_offset] = temp;
	  }
	}
      }
      Gadgetron::sum_over_dimension(cg_q_coil,cg_q,2);
      

      Gadgetron::ho2DArray< std::complex< float > > cg_x(coil_im_dims_[0],coil_im_dims_[1]);
      cg_x.fill(0);
      Gadgetron::ho2DArray< std::complex< float > > cg_r = cg_q;
      Gadgetron::ho2DArray< std::complex< float > > cg_p = cg_r;
      
      Gadgetron::ho2DArray< std::complex< float > > cg_conj_multiply (coil_im_dims_[0],coil_im_dims_[1]);
      Gadgetron::ho2DArray< std::complex< float > > cg_scale_multiply (coil_im_dims_[0],coil_im_dims_[1]);
      Gadgetron::ho2DArray< std::complex< float > > cg_temp (coil_im_dims_[0],coil_im_dims_[1]);
      
      Gadgetron::multiplyConj(cg_q,cg_q,cg_conj_multiply);
      std::complex<float> bl2_cx(0,0);
      for (int iter = 0; iter<cg_conj_multiply.get_number_of_elements(); iter++)
      {
	bl2_cx = bl2_cx + cg_conj_multiply[iter];
      }
      float rl2 = std::abs(bl2_cx);
      
      float rl2_new;
      std::complex<float> pl2;
      std::complex<float> a;
      float a_r;
      
      int cg_iter_num = 20;
      for (int icg = 0; icg < cg_iter_num; icg++){
	
	Gadgetron::multiplyConj(cg_r,cg_r,cg_conj_multiply);
	std::complex<float> rl2_cx(0,0);
	for (int iter = 0; iter<cg_conj_multiply.get_number_of_elements(); iter++)
	{
	  rl2_cx = rl2_cx + cg_conj_multiply[iter];
	}
	rl2 = std::abs(rl2_cx);
	
	/* E*p = F*S*p */
	cg_q.fill(0);
	for(int icoil = 0; icoil<num_coils_; icoil++){
	  
	  /* S*p */
	  size_t offset = 0;
	  size_t offset1 = 0;
	  size_t data_offset = 0;
	  for (int iy = 0; iy<coil_im_dims_[1]; iy++){
	    for (int ix = 0; ix<coil_im_dims_[0]; ix++){
	      offset = ix*coil_im_dims_[1]+iy;
	      offset1 = iy*coil_im_dims_[0] + ix;
	      data_offset = icoil*coil_im_dims_[0]*coil_im_dims_[1] + iy*coil_im_dims_[0] + ix;
	      
	      std::complex<float> temp = cg_p[offset1];
	      temp = temp*d_coilMap[data_offset];
	      
	      my_plan.f_hat[offset][0] = temp.real();
	      my_plan.f_hat[offset][1] = temp.imag();
	     
	    }
	  }
	
	  /* F*S*p */
	  nfft_trafo(&my_plan);
	  
	  /* F^H * F * S* p */
	  nfft_adjoint(&my_plan);
	  
	  /* S^H * F^H * F * S* p */
	  for (int iy = 0; iy<coil_im_dims_[1]; iy++){
	    for (int ix = 0; ix<coil_im_dims_[0]; ix++){
	      offset = ix*coil_im_dims_[1]+iy;
	      data_offset = icoil*coil_im_dims_[0]*coil_im_dims_[1] + iy*coil_im_dims_[0]+ix;
	      
	      std::complex<float> temp(my_plan.f_hat[offset][0],my_plan.f_hat[offset][1]);
	      temp = temp*std::conj(d_coilMap[data_offset]);
	      
	      cg_q_coil[data_offset] = temp;
	     
	    }
	  }
	  Gadgetron::sum_over_dimension(cg_q_coil,cg_q,2);
	   
	}
	
	Gadgetron::multiplyConj(cg_q,cg_p,cg_conj_multiply);
	std::complex<float> pl2(0,0);
	for (int iter = 0; iter<cg_conj_multiply.get_number_of_elements(); iter++)
	{
	  pl2 = pl2+cg_conj_multiply[iter];
	}
	
	a = rl2/pl2;
	
	cg_scale_multiply = cg_p;
	Gadgetron::scal(a,cg_scale_multiply);
	Gadgetron::add(cg_x,cg_scale_multiply, cg_temp);
	cg_x = cg_temp;
	
	cg_scale_multiply = cg_q;
	Gadgetron::scal(a,cg_scale_multiply);
	Gadgetron::subtract(cg_r, cg_scale_multiply, cg_temp);
	cg_r = cg_temp;
	
	Gadgetron::multiplyConj(cg_r,cg_r,cg_conj_multiply);
	std::complex<float> rl2_new_cx(0,0);
	for (int iter = 0; iter<cg_conj_multiply.get_number_of_elements(); iter++)
	{
	  rl2_new_cx = rl2_new_cx + cg_conj_multiply[iter];
	}
	rl2_new =std::abs(rl2_new_cx);

	a_r = rl2_new/rl2;
	cg_scale_multiply = cg_p;
	Gadgetron::scal(a_r,cg_scale_multiply);
	Gadgetron::add(cg_r,cg_scale_multiply, cg_temp);
	cg_p = cg_temp;
	
	std::cout<<"CG iter: "<<icg<<"  ||r||_2 = "<<rl2_new<<std::endl;
	if(std::abs(rl2_new) < 1e-5)
	  break;
	
      }
      
      //-------------------------------------------------------------------------//
      /* prepare image data array and header: m3->cont(m4) for sending out */
      // Prepare the data arrray containing estimated coil sensitivity map
      std::vector<size_t> im_dims_;
      im_dims_.clear();
      im_dims_.push_back(coil_im_dims_[0]);
      im_dims_.push_back(coil_im_dims_[1]);
      im_dims_.push_back(image_dimensions_[2]);
      
      for (int icoil = 0; icoil<num_coils_; icoil++){
	
	GadgetContainerMessage< hoNDArray< std::complex<float> > >* m4 = 
	new GadgetContainerMessage<hoNDArray< std::complex<float> > >();
	
	try{m4->getObjectPtr()->create(&im_dims_);}
	catch (std::runtime_error &err){
	  GEXCEPTION(err,"Unable to allocate new image array\n");
	  m4->release();
	  return -1;
	}
	std::complex<float>* d4 = m4->getObjectPtr()->get_data_ptr();
	
	
	size_t offset = 0;
	size_t data_offset = 0;
	//Copy the reconstructed image for all the channels
	for (int iy = 0; iy<coil_im_dims_[1]; iy++){
	  for (int ix = 0; ix<coil_im_dims_[0]; ix++){
	    offset = iy*coil_im_dims_[0]+ix;
	    data_offset = icoil*coil_im_dims_[0]*coil_im_dims_[1] + iy*coil_im_dims_[0]+ix;
	    d4[offset] = d_coilMap[data_offset];
	  }
	}
	
	// Prepare the image header
	GadgetContainerMessage<ISMRMRD::ImageHeader> *m3 = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
	ISMRMRD::AcquisitionHeader *base_head = m1->getObjectPtr();
	
	{
	  // Initialize header to all zeroes (there is a few fields we do not set yet)
	  ISMRMRD::ImageHeader tmp;
	  *(m3->getObjectPtr()) = tmp;
	}
	
	m3->getObjectPtr()->version = base_head->version;
	m3->getObjectPtr()->flags = base_head->flags;
	m3->getObjectPtr()->measurement_uid = base_head->measurement_uid;
	
	m3->getObjectPtr()->matrix_size[0] = im_dims_[0];
	m3->getObjectPtr()->matrix_size[1] = im_dims_[1];
	m3->getObjectPtr()->matrix_size[2] = im_dims_[2];
	
	m3->getObjectPtr()->field_of_view[0] = fov_[0];
	m3->getObjectPtr()->field_of_view[1] = fov_[1];
	m3->getObjectPtr()->field_of_view[2] = fov_[2];
	
	m3->getObjectPtr()->channels = 1;
	m3->getObjectPtr()->slice = base_head->idx.slice;
	m3->getObjectPtr()->set = icoil;
	
	m3->getObjectPtr()->acquisition_time_stamp = base_head->acquisition_time_stamp;
	memcpy(m3->getObjectPtr()->physiology_time_stamp, base_head->physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);
	
	memcpy(m3->getObjectPtr()->position, base_head->position, sizeof(float)*3);
	memcpy(m3->getObjectPtr()->read_dir, base_head->read_dir, sizeof(float)*3);
	memcpy(m3->getObjectPtr()->phase_dir, base_head->phase_dir, sizeof(float)*3);
	memcpy(m3->getObjectPtr()->slice_dir, base_head->slice_dir, sizeof(float)*3);
	memcpy(m3->getObjectPtr()->patient_table_position, base_head->patient_table_position, sizeof(float)*3);
	
	m3->getObjectPtr()->data_type = ISMRMRD::ISMRMRD_CXFLOAT;
	m3->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
	m3->getObjectPtr()->image_index = icoil+1; 
	m3->getObjectPtr()->image_series_index = 0;
	
	memcpy(m3->getObjectPtr()->user_int, m1->getObjectPtr()->user_int, sizeof(int32_t)*8);
	memcpy(m3->getObjectPtr()->user_float, m1->getObjectPtr()->user_float, sizeof(float)*8);
	
	
	
	/* Put header and data array on the recon stream */
	m3->cont(m4);
	
	if (this->next()->putq(m3) < 0){
	  GDEBUG("Failed to put coil images into message queue \n");
	  m3->release();
	  return GADGET_FAIL;
	}
      }
      
      //-------------------------------------------------------------------------//
      /* prepare image data array and header: m5->cont(m6) for sending out */
      // Prepare the data array containing the final reconstructed image
      GadgetContainerMessage< hoNDArray<std::complex<float> > >* m6 = 
      new GadgetContainerMessage< hoNDArray<std::complex<float> > >();
      
      try{m6->getObjectPtr()->create(&im_dims_);}
      catch (std::runtime_error &err){
	GEXCEPTION(err,"CombineGadget, failed to allocate new array\n");
	return -1;
      }
 
      std::complex<float>* d_sosim = m6->getObjectPtr()->get_data_ptr();
     
      int nz = 1;
      int ny = im_dims_[1];
      int nx = im_dims_[0];
      int nc = num_coils_;
      
      size_t img_block = nx*ny*nz;
      
      for (size_t z = 0; z < nz; z++) {
	for (size_t y = 0; y < ny; y++) {
	  for (size_t x = 0; x < nx; x++) {
	    size_t offset = z*ny*nx+y*nx+x;
// 	    d_sosim[offset] = d_accim[offset];
	    d_sosim[offset] = cg_x[offset];
	  }
	}
      }
      
      // Modify header to match the size and change the type to real
      // Prepare the image header
      GadgetContainerMessage<ISMRMRD::ImageHeader> *m5 = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
      ISMRMRD::AcquisitionHeader *base_head = m1->getObjectPtr();
      
      {
	// Initialize header to all zeroes (there is a few fields we do not set yet)
	ISMRMRD::ImageHeader tmp;
	*(m5->getObjectPtr()) = tmp;
      }
      
      m5->getObjectPtr()->version = base_head->version;
      m5->getObjectPtr()->flags = base_head->flags;
      m5->getObjectPtr()->measurement_uid = base_head->measurement_uid;
      
      m5->getObjectPtr()->matrix_size[0] = im_dims_[0];
      m5->getObjectPtr()->matrix_size[1] = im_dims_[1];
      m5->getObjectPtr()->matrix_size[2] = im_dims_[2];
      
      m5->getObjectPtr()->field_of_view[0] = fov_[0];
      m5->getObjectPtr()->field_of_view[1] = fov_[1];
      m5->getObjectPtr()->field_of_view[2] = fov_[2];
      
      m5->getObjectPtr()->channels = 1;
      m5->getObjectPtr()->slice = base_head->idx.slice;
      m5->getObjectPtr()->set = base_head->idx.set;
      
      m5->getObjectPtr()->acquisition_time_stamp = base_head->acquisition_time_stamp;
      memcpy(m5->getObjectPtr()->physiology_time_stamp, base_head->physiology_time_stamp, sizeof(uint32_t)*ISMRMRD::ISMRMRD_PHYS_STAMPS);
      
      memcpy(m5->getObjectPtr()->position, base_head->position, sizeof(float)*3);
      memcpy(m5->getObjectPtr()->read_dir, base_head->read_dir, sizeof(float)*3);
      memcpy(m5->getObjectPtr()->phase_dir, base_head->phase_dir, sizeof(float)*3);
      memcpy(m5->getObjectPtr()->slice_dir, base_head->slice_dir, sizeof(float)*3);
      memcpy(m5->getObjectPtr()->patient_table_position, base_head->patient_table_position, sizeof(float)*3);
      
      m5->getObjectPtr()->data_type = ISMRMRD::ISMRMRD_CXFLOAT;
      m5->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE;
      m5->getObjectPtr()->image_index = 1; 
      m5->getObjectPtr()->image_series_index = 1;
      
      memcpy(m5->getObjectPtr()->user_int, m1->getObjectPtr()->user_int, sizeof(int32_t)*8);
      memcpy(m5->getObjectPtr()->user_float, m1->getObjectPtr()->user_float, sizeof(float)*8);
      
      // Now add the new array to the outgoing message
      m5->cont(m6);
      
      return this->next()->putq(m5);
      
      m1->release();
    }
  }
  
  bool cpuRadialCGSENSEGadget::cal_radial2D_traj(size_t profiles_, size_t samples_per_profile_, std::vector<float>& kx, std::vector<float>& ky, std::vector<float>& dcf, int mode_)
  {
    try{

      int nSpokes = profiles_;
      int nFE = samples_per_profile_;
      arma::fvec fepoints = arma::linspace<arma::fvec>(-0.5, 0.5-1/float(nFE), nFE);
      arma::fvec a = arma::linspace<arma::fvec>(0, nSpokes-1, nSpokes);
      
      if (mode_ == 0)
      {
	if (nSpokes%2 == 0){
	  a = a*M_PI/float(nSpokes);
	}
	else{
	  a = a*M_PI*2/float(nSpokes);
	}
      }
      if (mode_ == 1)
      {
	a = a*(111.246*M_PI/180);
      }
      kx.clear();
      ky.clear();
      dcf.clear();
      size_t offset;
      for (int iSpokes = 0; iSpokes < nSpokes; iSpokes++){
	for (int iSamples = 0; iSamples < nFE; iSamples++){
	  kx.push_back(fepoints(iSamples)*cos(a(iSpokes)));
	  ky.push_back(fepoints(iSamples)*sin(a(iSpokes)));
	  dcf.push_back(abs(fepoints(iSamples))*nFE/nSpokes);
	}
      }
    }
    catch(...)
    {
      GERROR_STREAM("Errors in cpuRadialCGSENSEGadget::cal_radial2D_traj(...) ... ");
      return false;
    }
    return true;
  }
  
  bool cpuRadialCGSENSEGadget::cal_radial2D_coilKspace_to_coilIm(const Gadgetron::hoNDArray< std::complex<float> >* coilKspace, Gadgetron::hoNDArray< std::complex<float> >& coilIm, std::vector<size_t> coil_im_dims_, std::vector<float> kx, std::vector<float> ky, std::vector<float> dcf)
  {
    try{
      std::complex<float>* b = coilKspace->get_data_ptr();
      
      int profiles_ = coilKspace->get_size(1);
      int samples_per_profile_ = coilKspace->get_size(0);
      int num_coils_=coilKspace->get_size(2);
      
      size_t M = samples_per_profile_*profiles_;
      int weight = 0;
      double real,imag;           /* to read the real and imag part of a complex number */
      
      int my_N[2],my_n[2];         /* to init the nfft */
      int m = 6;
      double alpha = 2.0;
      int flags = PRE_PHI_HUT| PRE_PSI |MALLOC_X| MALLOC_F_HAT|
      MALLOC_F| FFTW_INIT| FFT_OUT_OF_PLACE|
      FFTW_MEASURE| FFTW_DESTROY_INPUT;
      
      my_N[0]=coil_im_dims_[0]; my_n[0]=std::ceil(my_N[0]*alpha);
      my_N[1]=coil_im_dims_[1]; my_n[1]=std::ceil(my_N[1]*alpha);
      
      
      /* initialise my_plan */
      nfft_plan my_plan;            /* plan for the two dimensional nfft  */
      nfft_init_guru(&my_plan, 2, my_N, M, my_n, m, flags, FFTW_MEASURE | FFTW_DESTROY_INPUT);
      
      /* read nonuniform (x,y) from the trajectory */
      int j = 0;
      for (int iky = 0; iky<profiles_; iky++){
	for (int ikx = 0; ikx<samples_per_profile_; ikx++){
	  j = iky*samples_per_profile_ + ikx;
	  my_plan.x[2*j+0] = kx[j];
	  my_plan.x[2*j+1] = ky[j];
	}
      }
      
      /* precompute psi */
      if(my_plan.flags & PRE_PSI)
	nfft_precompute_psi(&my_plan);
      
      /* precompute full psi */
      if(my_plan.flags & PRE_FULL_PSI)
	nfft_precompute_full_psi(&my_plan);
      
      coilIm.create(&coil_im_dims_);
      std::complex<float>* d_coilIm = coilIm.get_data_ptr();
      size_t data_offset = 0;
      for(int icoil = 0; icoil<num_coils_; icoil++){
	
	/* read freal and fimag from the knots */
	for (int iky = 0; iky<profiles_; iky++){
	  for (int ikx = 0; ikx<samples_per_profile_; ikx++){
	    data_offset = icoil*samples_per_profile_*profiles_ + iky*samples_per_profile_ + ikx;
	    j = iky*samples_per_profile_ + ikx;
	    
	    // densitivity compensation
	    my_plan.f[j][0] = b[data_offset].real()*dcf[j];
	    my_plan.f[j][1] = b[data_offset].imag()*dcf[j];
	    
	  }
	}
	
	nfft_adjoint(&my_plan);
	
	size_t offset = 0;
	//Copy the reconstructed image for all the channels
	for (int iy = 0; iy<coil_im_dims_[1]; iy++){
	  for (int ix = 0; ix<coil_im_dims_[0]; ix++){
	    offset = ix*coil_im_dims_[1]+iy;
	    data_offset = icoil*coil_im_dims_[0]*coil_im_dims_[1] + iy*coil_im_dims_[0]+ix;
	    std::complex<float> temp(my_plan.f_hat[offset][0],my_plan.f_hat[offset][1]);
	    d_coilIm[data_offset] = temp;
	  }
	}
      }
      
      /* finalize the nfft */
      nfft_finalize(&my_plan);
    }
    catch(...)
    {
      GERROR_STREAM("Errors in cpuRadialCGSENSEGadget::cal_radial2D_traj(...) ... ");
      return false;
    }
    return true;
  }
  
  GADGET_FACTORY_DECLARE(cpuRadialCGSENSEGadget)
}

