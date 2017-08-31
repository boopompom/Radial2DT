#ifndef Trajectory2D_h
#define Trajectory2D_h

#include <math.h>
#include <iostream>

#define MAX_PROJECTION_NUMBER (4096)

inline double ellipse(double angle, double x, double y)
{
	return ( 1.0 / sqrt( pow(cos(angle)/x,2.0) + pow(sin(angle)/y,2.0) ) );
}

inline void calc_2d_angles(long *nangles, double *theta, double fov1, double fov2, double theta_width, double res1 = 1.0, double res2 = 1.0 )
{
	int n = 0;
	double S;
	double del_theta_est, del_theta;
	const double HPI = M_PI_2;
	
	theta[n] = 0.0;
	
	while (theta[n] < theta_width) 
	{
		del_theta_est = 1.0 / ( ellipse(theta[n], 1.0 / (2.0*res1), 1.0 / (2.0*res2)) * 
			ellipse(theta[n]+ HPI, fov1, fov2) );

		del_theta = 1.0 / ( ellipse(theta[n] + del_theta_est/2.0 , 1.0 / (2.0*res1), 1.0 / (2.0*res2)) * 
			ellipse(theta[n] + del_theta_est/2.0 + HPI, fov1, fov2) );
		
		theta[n+1] = theta[n] + del_theta;
		
		n++;
		
	}
	
	/* adjust angles for symmetry based on which spoke is closer to theta_width */
	
	if ( theta[n] - theta_width > theta_width - theta[n-1] ) 
	{
		*nangles = n-1;
		S = theta_width / theta[n-1];
	} 
	else 
	{
		*nangles = n;
		S = theta_width / theta[n];
	}
	
	for (n = 0; n < *nangles; n++) 
	{
		theta[n] *= S;
		//kmax[n] = ellipse(theta[n], 1.0/(2.0*res1), 1.0/(2.0*res2));
		//dcf[n] = kmax[n] / ellipse(theta[n]+ HPI, fov1, fov2);
	}
	
	return;
}


namespace Gadgetron{
class Trajectory2D
{
public:
    
	// constructor
	Trajectory2D ();
    
	// destructor
	virtual ~Trajectory2D();
	
	enum eTrajectoryRange { UndefinedRange=0, FullRange, HalfRange };
	enum eTrajectoryType  { Uniform =0x12, GoldenAngle };
	
	
	bool initializeTrajectory( eTrajectoryRange eTrajRange, eTrajectoryType eTrajType, long lNumberOfProjections, long lFOV1, long lFOV2);
	
	bool calculateTrajectory();
	
	double getAzimuthalAngle         (long);
	long   getlFullSampleProjections (void);
	long   getlNumberOfProjections   (void);
	void   setlFibonacciN            (long);
	
protected:
	
	eTrajectoryRange m_eTrajectoryRange;  // Half sphere or full sphere
	eTrajectoryType  m_eTrajectoryType;
	long   m_lNumberOfProjections;        // Total Number Of Projections
	long   m_lFullSampleProjections;
	long   m_lFOV1;
	long   m_lFOV2;
	long   m_lFibonacciN;
	
	double *m_adAzimuthalAngle;           // x-y plane 

private:
    void calc2DGoldenAngle(double *theta, double fov1, double fov2, long projections_num, long fibonacci_N = 1);
	
    void calc2DUniform    (double *theta, double fov1, double fov2, double theta_width, long projections_num);
    
}; // class Trajectory

inline Trajectory2D::Trajectory2D ()
:   m_eTrajectoryRange          (UndefinedRange)                   
,   m_eTrajectoryType           (Uniform)
,   m_lNumberOfProjections      (0)
,   m_lFOV1                     (0)
,   m_lFOV2                     (0)
,   m_lFibonacciN               (1)
,   m_adAzimuthalAngle          (NULL)
{}

inline Trajectory2D::~Trajectory2D (void)
{
    delete[] m_adAzimuthalAngle;
	m_adAzimuthalAngle = NULL;
}

inline bool Trajectory2D::initializeTrajectory
(
 eTrajectoryRange eTrajRange,
 eTrajectoryType  eTrajType,
 long lNumberOfProjections,
 long lFOV1,
 long lFOV2
 )
{
	
    m_eTrajectoryRange       = eTrajRange;      
    m_eTrajectoryType        = eTrajType;
    m_lNumberOfProjections   = lNumberOfProjections;
    m_lFOV1                  = lFOV1;
    m_lFOV2                  = lFOV2;
    
    if (m_lNumberOfProjections > MAX_PROJECTION_NUMBER)
    {
        return false;
    }
		
    if( m_adAzimuthalAngle != NULL )
    {
	delete[] m_adAzimuthalAngle;
	m_adAzimuthalAngle = NULL;          
    }
       
    return true;
}

inline bool Trajectory2D::calculateTrajectory()
{
	//long lAngles = 0;
	double theta_width = M_PI;

	switch (m_eTrajectoryType)
	{
	case Uniform:
		{
			double* dAzimuthalAngle = new double [MAX_PROJECTION_NUMBER];
			calc_2d_angles( &m_lFullSampleProjections, dAzimuthalAngle,  m_lFOV1, m_lFOV2, theta_width );

			if ( m_lNumberOfProjections >= m_lFullSampleProjections)
				m_lNumberOfProjections = m_lFullSampleProjections;
			else
				calc2DUniform(dAzimuthalAngle, m_lFOV1, m_lFOV2, theta_width, m_lNumberOfProjections );	
			
			if( m_adAzimuthalAngle != NULL )
			{
				delete[] m_adAzimuthalAngle;
				m_adAzimuthalAngle = NULL;          
			}
			m_adAzimuthalAngle = new double [m_lNumberOfProjections];
			if( m_adAzimuthalAngle == NULL )
				return false; 	
			
			if ( m_lNumberOfProjections%2 == 0)
			{
				for (int i = 0; i < m_lNumberOfProjections/2; i ++)
				{
					m_adAzimuthalAngle[i] = dAzimuthalAngle[i*2];
					m_adAzimuthalAngle[i+m_lNumberOfProjections/2] = dAzimuthalAngle[i*2+1] + M_PI;
				}
			}
			else
			{
				for (int i = 0; i < (int)(m_lNumberOfProjections/2); i ++)
				{
					m_adAzimuthalAngle[i] = dAzimuthalAngle[i*2];
					m_adAzimuthalAngle[i+(int)(m_lNumberOfProjections/2)+1] = dAzimuthalAngle[i*2+1] + M_PI;
				}
				m_adAzimuthalAngle[(int)(m_lNumberOfProjections/2)] = dAzimuthalAngle[m_lNumberOfProjections-1];			
			}
			
			delete[] dAzimuthalAngle;
			break;
		}
	case GoldenAngle:	
		{
			if( m_adAzimuthalAngle != NULL )
			{
				delete[] m_adAzimuthalAngle;
				m_adAzimuthalAngle = NULL;          
			}
			m_adAzimuthalAngle = new double [m_lNumberOfProjections];
			calc2DGoldenAngle(m_adAzimuthalAngle, m_lFOV1, m_lFOV2, m_lNumberOfProjections, m_lFibonacciN );
			break;
		}
	default:
		{
			return (false);
			break;
		}
	}

	return (true);
}

inline double Trajectory2D::getAzimuthalAngle(long lLine)
{
	return m_adAzimuthalAngle[MIN(lLine, m_lNumberOfProjections - 1)];
}

inline long Trajectory2D::getlFullSampleProjections(void)
{
	return m_lFullSampleProjections;
}

inline long Trajectory2D::getlNumberOfProjections(void)
{
	return m_lNumberOfProjections;
}

inline void Trajectory2D::setlFibonacciN (long lFibonacciN)
{
	m_lFibonacciN = lFibonacciN;	
}

inline void Trajectory2D::calc2DUniform(double *theta, double fov1, double fov2, double theta_width, long projections_num)
{
	long lI = 0;
	double dSubSpace = theta_width/(double)(projections_num-1);

	theta[0] = 1.0/ellipse(0.0, fov2/2, fov1/2);

	for (lI = 1; lI < projections_num; lI ++ )
	{
		double theta_org = lI * dSubSpace;
		double dtheta_map = 1.0/ellipse(theta_org, fov2/2, fov1/2);

		theta[lI] = dtheta_map + theta[lI-1];
	}
	for (lI = 0; lI < projections_num; lI ++ )
	{
		theta[lI] = theta[lI]*theta_width/theta[projections_num-1];
	}

}

inline void Trajectory2D::calc2DGoldenAngle(double *theta, double fov1, double fov2, long projections_num, long fibonacci_N )
{
	long lI;
	const long Theta_Map_Num = 1e4;
	double* theta_param = new double[Theta_Map_Num];
	
	calc2DUniform(theta_param, fov1, fov2, 2*M_PI, Theta_Map_Num);

	double dSubSpace = 2*M_PI/(double)(Theta_Map_Num-1);

	const double d_golden_ratio = 1.61803398875; 
	const double d_golden_angle = M_PI/(double)(d_golden_ratio + fibonacci_N - 1);

	theta[0] = 0;
	for(lI = 1; lI < projections_num; lI ++) 
	{
		double theta_isotropic = fmod(d_golden_angle*lI, 2*M_PI);
		long   lInd =  static_cast<long>(theta_isotropic/dSubSpace + 0.5);
	    theta[lI] = theta_param[lInd];

	}
	
	delete[] theta_param;

}
}
#endif

