#include "gpuRadialCGSENSEGadget.h"

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

//-----------------------------------------------//
// cgsense reconstruction using GPU
// implemented by Jia Sen on 2016/01/14
// Bug: the reconstructed image matrix is square
//       not reduced FOV reconstruction
//      This may induce error when online recon
//-----------------------------------------------//


namespace Gadgetron{
  
  gpuRadialCGSENSEGadget::gpuRadialCGSENSEGadget()
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
  , oversampling_factor_(1.25)
  , traj_mode(Trajectory2D::Uniform)
  {
  }
  
  gpuRadialCGSENSEGadget::~gpuRadialCGSENSEGadget()
  {
    if(buffer_) delete buffer_;
  }
  
  int gpuRadialCGSENSEGadget::process_config(ACE_Message_Block* mb)
  {
    //GDEBUG("gpuRadialCGSENSEGadget::process_config\n");
    
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
    
//     this->cal_radial2D_dcf (dcf, image_dimensions_recon_[0], image_dimensions_recon_[1]);
    
    GDEBUG("encoding limits slices_: %d, profiles_: %d \n", slices_, profiles_);
    
    
    return GADGET_OK;
  }
  
  int gpuRadialCGSENSEGadget::
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
      this->cal_radial2D_dcf(dcf,samples_per_profile_,profiles_,image_dimensions_recon_[0], image_dimensions_recon_[1]);
      
      
      //---------------------------------------------------------------------//
      // 2DT reconstruction
      // profiles_per_frame_ = profiles_ will reconstruct only one image. 
      //--------------------------------------------------------------------//
      profiles_per_frame_ = long(profiles_*0.5);
      frames_per_reconstruction_ = long (profiles_/profiles_per_frame_);
      size_t samples_per_frame_ = profiles_per_frame_*samples_per_profile_;
      
      std::vector<float> dcw_per_frame;
      this->cal_radial2D_dcf(dcw_per_frame,samples_per_profile_,profiles_per_frame_,image_dimensions_recon_[0], image_dimensions_recon_[1]);
      
      std::cout<<"--------------------------------------------------------------"<<std::endl;
      std::cout<< "All profiles are accumulated, we are ready to reconstruct !"<<std::endl;
      std::cout<<"Profiles:"<<profiles_<<"  Slices:"<<slices_<< std::endl;
      std::cout<<"Samples_per_profiles:"<<samples_per_profile_<<" Profiles_per_frame:"<< profiles_per_frame_ <<"  frames_per_reconstruction:"<< frames_per_reconstruction_ <<std::endl;
      std::cout<<"--------------------------------------------------------------"<<std::endl;
      
      
      // Configuration image dimensions_
      uint64d2 matrix_size = uint64d2(image_dimensions_recon_[0],image_dimensions_recon_[1]);
      uint64d2 matrix_size_os = uint64d2(image_dimensions_recon_os_[0],image_dimensions_recon_os_[1]);
      
      _real kernel_width = 5.5;
      _real alpha = (_real)matrix_size_os.vec[0]/(_real)matrix_size.vec[0];
      
      //-------------------------------//
      // CGSENSE on GPU
      //-------------------------------//
 
      GPUTimer *timer;
      cuNFFT_plan<_real,2> plan( matrix_size, matrix_size_os, kernel_width );
      
    
      // slice-by-slice reconstruction
      for(unsigned int islice = 0; islice<slices_; islice++){
	
	//-----------------------------//
	// Estimate Coil Sensitivity coilMap
	// by gridding all profiles to 
	// one image
	//-----------------------------//
	
	timer = new GPUTimer("GPU gridding and FFT");
	std::vector<size_t> csm_dim_array;
	csm_dim_array.push_back(samples_per_profile_*profiles_);
	csm_dim_array.push_back(1);
	boost::shared_ptr< hoNDArray<_complext> > host_samples_full ( new hoNDArray<_complext>(&csm_dim_array));
	boost::shared_ptr< hoNDArray<_reald2> >   host_traj_full    ( new hoNDArray<_reald2>(&csm_dim_array) );
	boost::shared_ptr< hoNDArray<_real> >     host_dcw_full     ( new hoNDArray<_real>(&csm_dim_array) );
	Gadgetron::hoNDArray< std::complex<float> > coilIm;	coilIm.create(image_dimensions_recon_[0],image_dimensions_recon_[1],num_coils_);
	std::complex<float>* d_coilIm = coilIm.get_data_ptr();

	// traj
	memcpy(host_traj_full->get_data_ptr(), all_traj->get_data_ptr(), sizeof(_reald2)*samples_per_profile_*profiles_);
	
	// dcw
	for (unsigned int iter_sample = 0; iter_sample<samples_per_profile_*profiles_; iter_sample++){
	  host_dcw_full->get_data_ptr()[iter_sample] = dcf[iter_sample];
	}
	
	// image
	vector<size_t> coilImage_dims = to_std_vector(matrix_size); 
	coilImage_dims.push_back(num_coils_);
	cuNDArray<_complext> coilImage_cu(&coilImage_dims);
	clear(&coilImage_cu);
	vector<size_t> image_dims_full = to_std_vector(matrix_size);
	for (unsigned int iter_coil = 0; iter_coil<num_coils_; iter_coil++){
	  // samples
	  size_t src_offset = islice*buffer_dims_[0]*buffer_dims_[1]*buffer_dims_[2]+iter_coil*buffer_dims_[0]*buffer_dims_[1];
	  memcpy(host_samples_full->get_data_ptr(), b+src_offset, sizeof(_complext)*samples_per_profile_*profiles_);
	  
	  cuNDArray<_complext> _samples(host_samples_full.get());
	  cuNDArray<_reald2> _trajectory(host_traj_full.get());
	  cuNDArray<_real> dcw(host_dcw_full.get());
	  
	  // Preprocess
	  plan.preprocess( &_trajectory, plan_type::NFFT_PREP_NC2C );
	  
	  cuNDArray<_complext> tmp_image(&image_dims_full, coilImage_cu.get_data_ptr()+iter_coil*prod(matrix_size));
	  
	  // Gridder
	  plan.compute( &_samples, &tmp_image, &dcw, plan_type::NFFT_BACKWARDS_NC2C );
	  
	} // end coil-by-coil gridding
	
	// Output result: from device to host
	boost::shared_ptr< hoNDArray<_complext> > coilImage_ho = coilImage_cu.to_host();
	memcpy(d_coilIm,coilImage_ho->get_data_ptr(),sizeof(_complext)*prod(matrix_size)*num_coils_);
	write_nd_array<_complext>(coilImage_ho.get(),"coilImage_ho.cplx");
	delete timer;
	
	
	//-------------------------------------------//
	// csm estimation
	//-------------------------------------------//
	Gadgetron::gtPlus::gtPlusISMRMRDReconUtilComplex<std::complex<float> > gtPlus_util_complex_;
	Gadgetron::ISMRMRDCOILMAPALGO algo = Gadgetron::ISMRMRD_SOUHEIL;
	unsigned long long csm_kSize_ = 7;
	unsigned long long csm_powermethod_num_ = 3;
	Gadgetron::hoNDArray< std::complex<float> > coilMap;  
	gtPlus_util_complex_.coilMap2DNIH(coilIm, coilMap, algo);
	write_nd_array<std::complex<float> >(&coilMap,"coilMap.cplx");
	  
	//-----------------------------------------------
	//--------------------------------------------//
	// prepare data
	std::vector<size_t> dim_array;
	dim_array.push_back(samples_per_profile_);
	dim_array.push_back(profiles_);
	dim_array.push_back(num_coils_);
	boost::shared_ptr< hoNDArray<_complext> > host_data ( new hoNDArray<_complext>(&dim_array));
	size_t b_offset = islice*samples_per_profile_*profiles_*num_coils_ ;
        memcpy(host_data->get_data_ptr(),b+b_offset,sizeof(_complext)*samples_per_profile_*profiles_*num_coils_);
	
	std::vector<size_t> dcw_dim_array;
	dcw_dim_array.push_back(samples_per_frame_);
        boost::shared_ptr< hoNDArray<_real> >     host_dcw  ( new hoNDArray<_real>(&dcw_dim_array) );
	for (unsigned int iter_sample = 0; iter_sample<samples_per_frame_; iter_sample++){
	  host_dcw->get_data_ptr()[iter_sample] = dcw_per_frame[iter_sample];
	}
	
	
	// Configuration from the host data
	unsigned int samples_per_profile = host_data->get_size(0);
	unsigned int num_profiles = host_data->get_size(1);
	unsigned int num_coils = host_data->get_size(2);
	
	uint64d2 matrix_size = uint64d2(image_dimensions_recon_[0],image_dimensions_recon_[1]);
	uint64d2 matrix_size_os = uint64d2(image_dimensions_recon_os_[0],image_dimensions_recon_os_[1]);
	
	_real kernel_width = 5.5;
	_real alpha = (_real)matrix_size_os.vec[0]/(_real)matrix_size.vec[0];
	_real kappa = 0.3;
	
	unsigned int num_iterations = 20;
	unsigned int profiles_per_frame = profiles_per_frame_;
	unsigned int frames_per_reconstruction = frames_per_reconstruction_;
	
	// Silent correction of invalid command line parameters (clamp to valid range)
	if( profiles_per_frame > num_profiles ) profiles_per_frame = num_profiles;
	if( frames_per_reconstruction < 0 ) frames_per_reconstruction = num_profiles / profiles_per_frame;
	if( frames_per_reconstruction*profiles_per_frame > num_profiles ) frames_per_reconstruction = num_profiles / profiles_per_frame;
	
	unsigned int profiles_per_reconstruction = frames_per_reconstruction*profiles_per_frame;
	unsigned int samples_per_frame = profiles_per_frame*samples_per_profile;
	unsigned int samples_per_reconstruction = profiles_per_reconstruction*samples_per_profile;
	
	cout << endl << "#samples/profile: " << samples_per_profile;
	cout << endl << "#profiles/frame: " << profiles_per_frame;
	cout << endl << "#profiles: " << num_profiles;
	cout << endl << "#coils: " << num_coils;
	cout << endl << "#frames/reconstruction: " << frames_per_reconstruction;
	cout << endl << "#profiles/reconstruction: " << profiles_per_reconstruction;
	cout << endl << "#samples/reconstruction: " << samples_per_reconstruction << endl << endl;
	
	// Set density compensation weights

	vector<size_t> dcw_dims; 
	dcw_dims.push_back( samples_per_frame ); 
	dcw_dims.push_back( 1 );
        boost::shared_ptr< cuNDArray<_real> > dcw ( new cuNDArray<_real>(&dcw_dims) );
	cudaMemcpy(dcw->get_data_ptr(), host_dcw->get_data_ptr(),
		   samples_per_frame*sizeof(_real),cudaMemcpyHostToDevice);
	
	// Define encoding matrix for non-Cartesian SENSE
	 const bool use_atomics = false;
	boost::shared_ptr< cuNonCartesianSenseOperator<_real,2,use_atomics> > E
	( new cuNonCartesianSenseOperator<_real,2,use_atomics>() );  
	
	E->setup( matrix_size, matrix_size_os, kernel_width );
	
	// Define rhs buffer
	boost::shared_ptr< cuSenseBuffer<_real,2,use_atomics> > rhs_buffer
	( new cuSenseBuffer<_real,2,use_atomics>() );
	
	rhs_buffer->setup( matrix_size, matrix_size_os, kernel_width, num_coils, 8, 16 );
	rhs_buffer->set_dcw(dcw);
	
	// Fill rhs buffer
	timer = new GPUTimer("Filling rhs buffer");
	
	// Go through all the data...
	for( unsigned int iteration = 0; iteration < num_profiles/profiles_per_frame; iteration++ ) {
	  
	  // Define trajectories
	  vector<size_t> traj_dims; 
	  traj_dims.push_back( samples_per_frame ); 
	  traj_dims.push_back( 1 );
          boost::shared_ptr< cuNDArray<_reald2> > co( new cuNDArray<_reald2>(&traj_dims) );
	  cudaMemcpy(co->get_data_ptr(),all_traj->get_data_ptr()+iteration*samples_per_frame,samples_per_frame*sizeof(_reald2),cudaMemcpyHostToDevice);
	  
// 	  boost::shared_ptr< cuNDArray<_reald2> > traj = compute_radial_trajectory_golden_ratio_2d<_real>
// 	  ( samples_per_profile, profiles_per_frame, 1, iteration*profiles_per_frame );
// 	  
	  // Upload data
	  boost::shared_ptr< cuNDArray<_complext> > csm_data = this->upload_data
	  ( iteration, samples_per_frame, num_profiles*samples_per_profile, num_coils, host_data.get() );
	  
	  // Add frame to rhs buffer
	  rhs_buffer->add_frame_data( csm_data.get(), co.get() );
	}
	
	delete timer;
	
	
	// Estimate CSM
	//
	
	timer = new GPUTimer("Estimating csm");
	
	boost::shared_ptr< cuNDArray<_complext> > acc_images = rhs_buffer->get_accumulated_coil_images();
	boost::shared_ptr< cuNDArray<_complext> > csm = estimate_b1_map<_real,2>( acc_images.get() ); 
	
	//---------------------------------//
// 	vector<size_t> coilMap_dims; 
// 	coilMap_dims.push_back(matrix_size[0]); 
// 	coilMap_dims.push_back(matrix_size[1]);
// 	coilMap_dims.push_back(num_coils);
//         boost::shared_ptr< cuNDArray<_complext> > csm ( new cuNDArray<_complext>(&coilMap_dims) );
// 	cudaMemcpy(csm->get_data_ptr(), coilMap.get_data_ptr(), prod(matrix_size)*num_coils_,cudaMemcpyHostToDevice);
	
	
	
	E->set_csm(csm);
	
	boost::shared_ptr< hoNDArray<_complext> > csm_ho = csm->to_host();
	write_nd_array<_complext>(csm_ho.get(),"csm_ho.cplx");
	
	
	delete timer;
	
	
	// Define regularization image operator 
	//
	
	timer = new GPUTimer("Computing regularization");
	
	std::vector<size_t> image_dims = to_std_vector(matrix_size);
	cuNDArray<_complext> reg_image = cuNDArray<_complext>(&image_dims);
	
	E->mult_csm_conj_sum( acc_images.get(), &reg_image );
	acc_images.reset();
	
	
	boost::shared_ptr< cuImageOperator<_complext> > R( new cuImageOperator<_complext>() );
	R->set_weight( kappa );
	R->compute( &reg_image );
	
	delete timer;
	
	// Define preconditioning weights
	//
	
	timer = new GPUTimer("Computing preconditioning weights");
	
	boost::shared_ptr< cuNDArray<_real> > _precon_weights = sum(abs_square(csm.get()).get(),2);
	boost::shared_ptr< cuNDArray<_real> > R_diag = R->get();
	*R_diag *= kappa;
	*_precon_weights += *R_diag;
	R_diag.reset();
	reciprocal_sqrt_inplace(_precon_weights.get());
	boost::shared_ptr< cuNDArray<_complext> > precon_weights = real_to_complex<_complext>( _precon_weights.get() );
	_precon_weights.reset();
	
	// Define preconditioning matrix
	boost::shared_ptr< cuCgPreconditioner<_complext> > D( new cuCgPreconditioner<_complext>() );
	D->set_weights( precon_weights );
	precon_weights.reset();
	csm.reset();
	
	delete timer;
	
	// 
	// Setup radial SENSE reconstructions
	//
	// Notify encoding operator of dcw
	sqrt_inplace(dcw.get());
	E->set_dcw(dcw);
	// Setup conjugate gradient solver
	cuCgSolver<_complext> cg;
	cg.set_preconditioner ( D );  // preconditioning matrix
	cg.set_max_iterations( num_iterations );
	cg.set_tc_tolerance( 1e-6 );
	cg.set_output_mode( cuCgSolver< _complext>::OUTPUT_VERBOSE );
	cg.set_encoding_operator( E );        // encoding matrix
	cg.add_regularization_operator( R );  // regularization matrix
	
	// Reconstruct all SENSE frames iteratively
	unsigned int num_reconstructions = num_profiles / profiles_per_reconstruction;
	
	// Allocate space for result
	image_dims.push_back(frames_per_reconstruction*num_reconstructions); 
	cuNDArray<_complext> result = cuNDArray<_complext>(&image_dims);
	
	timer = new GPUTimer("Full SENSE reconstruction.");
	
	// Define image dimensions
	image_dims = to_std_vector(matrix_size); 
	image_dims.push_back(frames_per_reconstruction);
	
	for( unsigned int reconstruction = 0; reconstruction<num_reconstructions; reconstruction++ ){
	  
	  // Determine trajectories
// 	  boost::shared_ptr< cuNDArray<_reald2> > traj = compute_radial_trajectory_golden_ratio_2d<_real>
// 	  ( samples_per_profile, profiles_per_frame, frames_per_reconstruction, reconstruction*profiles_per_reconstruction );
	  
	  // Define trajectories
	  vector<size_t> traj_dims; 
	  traj_dims.push_back( samples_per_frame ); 
	  traj_dims.push_back( frames_per_reconstruction);
          boost::shared_ptr< cuNDArray<_reald2> > traj( new cuNDArray<_reald2>(&traj_dims) );
	  cudaMemcpy(traj->get_data_ptr(), all_traj->get_data_ptr()+reconstruction*samples_per_reconstruction,samples_per_reconstruction*sizeof(_reald2),cudaMemcpyHostToDevice);
	  

	  
	  // Upload data
	  boost::shared_ptr< cuNDArray<_complext> > data = this->upload_data
	  ( reconstruction, samples_per_reconstruction, num_profiles*samples_per_profile, num_coils, host_data.get() );
	  
	  // Pass image dimensions to encoding operator
	  E->set_domain_dimensions(&image_dims);
	  E->set_codomain_dimensions(data->get_dimensions().get());
	  
	  // Set current trajectory and trigger NFFT preprocessing
	  E->preprocess(traj.get());
	  
	  *data *= *dcw;
	  //
	  // Invoke conjugate gradient solver
	  //
	  
	  boost::shared_ptr< cuNDArray<_complext> > cgresult;
	  {
	    GPUTimer timer("GPU Conjugate Gradient solve");
	    cgresult = cg.solve(data.get());
	  }
	  
	  if( !cgresult.get() )
	    return 1;
	  
	  // Copy cgresult to overall result
	  cuNDArray<_complext> out(&image_dims, result.get_data_ptr()+reconstruction*prod(matrix_size)*frames_per_reconstruction );    
	  out = *(cgresult.get());
	}
	
	delete timer;
	
	// All done, write out the result
	
	timer = new GPUTimer("Writing out result");
	boost::shared_ptr< hoNDArray<_complext> > host_result = result.to_host();
	delete timer;
	
	for(unsigned int iframe = 0; iframe<frames_per_reconstruction_; iframe++){
	  // Create a new message with an hoNDArray for the combined image
	  GadgetContainerMessage< hoNDArray<std::complex<float> > >* m8 = 
	  new GadgetContainerMessage< hoNDArray<std::complex<float> > >();
	  m8->getObjectPtr()->create(image_dimensions_recon_[0],image_dimensions_recon_[1]);
	  std::complex<float>* d_sosim = m8->getObjectPtr()->get_data_ptr();
	  
	  // 2D image
	  int ny = image_dimensions_recon_[1];
	  int nx = image_dimensions_recon_[0];
	  memcpy(d_sosim,host_result->get_data_ptr()+iframe*ny*nx,ny*nx*sizeof(std::complex<float>));
	  
	  
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
	  
	} // end of inner iframe loop for sending out image
      
    } // End of outer slice loop for slice-by-slice reconstruction
    
  } // End of is_allprofiles_loaded for whole reconstruction work
  
  m1->release();
  return GADGET_OK;
}

bool gpuRadialCGSENSEGadget::cal_radial2D_dcf ( std::vector<float>& dcf, size_t dcf_samples_per_profile_, size_t dcf_profiles_per_frame_, double fov1, double fov2)
{
  size_t nProfiles = dcf_profiles_per_frame_; 
  size_t nSamples  = dcf_samples_per_profile_;
  
  arma::fvec fepoints = arma::linspace<arma::fvec>(-0.5, 0.5-1/float(nSamples), nSamples);  
  for (size_t p = 0; p < nProfiles; p ++){
    for (size_t iSamples = 0; iSamples < samples_per_profile_; iSamples++){
      //dcf.push_back(abs(fepoints(iSamples))*samples_per_profile_/nProfiles);
      dcf.push_back(abs(fepoints(iSamples))*ellipse(traj.getAzimuthalAngle(p) + M_PI/2, fov2, fov1));
    }
  }
  
  return true;
}

// Upload samples for one reconstruction from host to device
boost::shared_ptr< cuNDArray<_complext> > 
gpuRadialCGSENSEGadget::upload_data( unsigned int reconstruction, unsigned int samples_per_reconstruction, unsigned int total_samples_per_coil, unsigned int num_coils, hoNDArray<_complext> *host_data )
{
  vector<size_t> dims; dims.push_back(samples_per_reconstruction); dims.push_back(num_coils);
  cuNDArray<_complext> *data = new cuNDArray<_complext>(); data->create( &dims );
  for( unsigned int i=0; i<num_coils; i++ )
    cudaMemcpy( data->get_data_ptr()+i*samples_per_reconstruction, 
		host_data->get_data_ptr()+i*total_samples_per_coil+reconstruction*samples_per_reconstruction, 
		samples_per_reconstruction*sizeof(_complext), cudaMemcpyHostToDevice );

  return boost::shared_ptr< cuNDArray<_complext> >(data);
}


GADGET_FACTORY_DECLARE(gpuRadialCGSENSEGadget)
}

