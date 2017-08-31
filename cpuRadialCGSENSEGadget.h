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



namespace Gadgetron{

  class EXPORTGADGETS_CPURADIAL cpuRadialCGSENSEGadget :
    public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
  {

  public:

    cpuRadialCGSENSEGadget();
    virtual ~cpuRadialCGSENSEGadget();

  protected:

    virtual int process_config(ACE_Message_Block *mb);

    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader > *m1,
			GadgetContainerMessage< hoNDArray< std::complex<float> > > *m2);

    int slices_;
    int sets_;
    long profiles_;
    long samples_per_profile_;
    unsigned int num_coils_;
    int image_counter_;
    
    int mode_;     // mode_ = 0 is fixed radial; mode_ = 1 is golden radial
    
    hoNDArray< std::complex<float> >* buffer_;
    std::vector<size_t> buffer_dims_;

    // Internal book-keping
    boost::shared_array<long> previous_profile_;
    boost::shared_array<long> profiles_counter_frame_;
    boost::shared_array<long> profiles_counter_global_;

    float kernel_width_;
    float oversampling_factor_;

    
    boost::shared_array< hoNDArray<floatd2> > host_traj_recon_;
    boost::shared_array< hoNDArray<float> > host_weights_recon_;
    

    std::vector<size_t> fov_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> image_dimensions_;
    
    
    bool cal_radial2D_traj(size_t profiles_, size_t samples_per_profile_, std::vector<float>& kx, std::vector<float>& ky, std::vector<float>& dcf, int mode_);
    
    bool cal_radial2D_coilKspace_to_coilIm(const Gadgetron::hoNDArray< std::complex<float> >* coilKspace, Gadgetron::hoNDArray< std::complex<float> >& coilIm, std::vector<size_t> coil_im_dims_, std::vector<float> kx, std::vector<float> ky, std::vector<float> dcf);
    

  private:
    

    inline bool vec_equal(float *in1, float *in2) {
      for (unsigned int i = 0; i < 3; i++) {
        if (in1[i] != in2[i]) return false;
      }
      return true;
    }   
  };
}
