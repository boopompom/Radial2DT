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

#define NFFT_PRECISION_DOUBLE
#include "nfft3.h"

#include "Trajectory2D.h"


namespace Gadgetron{

  class EXPORTGADGETS_CPURADIAL cpuRadialNUFFTGadget :
    public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
  {

  public:

    cpuRadialNUFFTGadget();
    virtual ~cpuRadialNUFFTGadget();

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
    
    
    hoNDArray< std::complex<float> >* buffer_;
    std::vector<size_t> buffer_dims_;

    std::vector<size_t> fov_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> image_dimensions_;
    std::vector<float> slice_positions;
   

    
    std::vector<float> dcf;
    bool construct_nfft_plan(nfft_plan& my_plan, long iframe);
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
