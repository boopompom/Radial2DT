#ifndef GADGETRON_CPURADIAL_EXPORT_H_
#define GADGETRON_CPURADIAL_EXPORT_H_

#if defined (WIN32)
#if defined (__BUILD_GADGETRON_CPURADIAL__)
#define EXPORTGADGETS_CPURADIAL __declspec(dllexport)
#else
#define EXPORTGADGETS_CPURADIAL __declspec(dllimport)
#endif
#else
#define EXPORTGADGETS_CPURADIAL
#endif

#endif /* GADGETRON_CPURADIAL_EXPORT_H_ */
