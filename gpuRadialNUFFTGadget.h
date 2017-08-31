#pragma once

#include "gadgetron_cpuradial_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"
#include "vector_td.h"


#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/concept_check.hpp>

#include <armadillo>
#define ARMA_64BIT_WORD

// GPU NFFT include
#include "cuNDArray_elemwise.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "hoNDArray_fileio.h"
#include "vector_td_utilities.h"
#include "cuImageOperator.h"
#include "radial_utilities.h"
#include "cuNonCartesianSenseOperator.h"
#include "cuSenseBuffer.h"
#include "cuCgPreconditioner.h"
#include "cuCgSolver.h"
#include "b1_map.h"
#include "parameterparser.h"
#include "GPUTimer.h"

#include "Trajectory2D.h"

using namespace std;
using namespace Gadgetron;

// Define desired precision
typedef float _real; 
typedef complext<_real> _complext;
typedef reald<_real,2>::Type _reald2;
typedef cuNFFT_plan<_real,2> plan_type;


namespace Gadgetron{

  class EXPORTGADGETS_CPURADIAL gpuRadialNUFFTGadget :
    public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
  {

  public:

    gpuRadialNUFFTGadget();
    virtual ~gpuRadialNUFFTGadget();

  protected:

    virtual int process_config(ACE_Message_Block *mb);

    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader > *m1,
			GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2);

    int slices_;
    long profiles_;
    long samples_per_profile_;
    unsigned int num_coils_;
    int image_counter_;
    long profiles_per_frame_;
    long frames_per_reconstruction_;
    
       std::vector<float> slice_positions;
    
    hoNDArray< std::complex<float> >* buffer_;
    std::vector<size_t> buffer_dims_;

    std::vector<size_t> fov_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> image_dimensions_;
    std::vector<size_t> image_dimensions_original_;
    std::vector<size_t> image_dimensions_recon_;
    uint64d2 image_dimensions_recon_os_;
    
    float reconstruction_os_factor_x = 1.0;
    float reconstruction_os_factor_y = 1.0;
    float kernel_width_ = 5.5;
    float oversampling_factor_ = 2;
  
    std::vector<float> dcf;
    
    bool cal_radial2D_dcf (std::vector<float>& dcf, double fov1, double fov2);
    Trajectory2D traj;
    Trajectory2D::eTrajectoryType  traj_mode;
   
  private:
    

    inline bool vec_equal(float *in1, float *in2) {
      for (unsigned int i = 0; i < 3; i++) {
        if (in1[i] != in2[i]) return false;
      }
      return true;
    }   
  };
}
