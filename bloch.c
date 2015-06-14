#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

///////////////
// CONSTANTS //
///////////////

#define DEBUG 1
#define WRITE_FID 0
#define WRITE_FID_ZERO_CROSSINGS 0
#define WRITE_FID_FREQUENCIES 1
#define WRITE_PI2 0

#define GAMMA_P (2.675222005e8)

/////////////////////////
// CVODE HELPER MACROS //
/////////////////////////

#define X(v,i) NV_Ith_S(v,3*i)
#define Y(v,i) NV_Ith_S(v,3*i + 1)
#define Z(v,i) NV_Ith_S(v,3*i + 2)

#define XX(A,i) DENSE_ELEM(A,i,i)
#define XY(A,i) DENSE_ELEM(A,i,i+1)
#define XZ(A,i) DENSE_ELEM(A,i,i+2)
#define YX(A,i) DENSE_ELEM(A,i+1,i)
#define YY(A,i) DENSE_ELEM(A,i+1,i+1)
#define YZ(A,i) DENSE_ELEM(A,i+1,i+2)
#define ZX(A,i) DENSE_ELEM(A,i+2,i)
#define ZY(A,i) DENSE_ELEM(A,i+2,i+1)
#define ZZ(A,i) DENSE_ELEM(A,i+2,i+2)

struct bloch_sim {
    int rf_on;

    realtype w_0;
    realtype B_0;
    realtype w_1;
    realtype B_1;

    realtype T_1;
    realtype T_2;

    int num_cells;
    realtype *cell_frequencies;
    realtype *cell_positions;

    realtype w_avg;
    realtype B_avg;

    realtype p_0;
    realtype p_1;

    int fid_num_samples;
    int fid_samples_per_period;
    realtype fid_duration;
    realtype fid_sampling_interval;
    realtype fid_sampling_rate;

    int pi2_num_samples;
    int pi2_samples_per_period;
    realtype pi2_duration;
    realtype pi2_sampling_interval;
    realtype pi2_sampling_rate;

    int num_zero_crossings;
    realtype *zero_crossings;
};

static void initialize_bloch(struct bloch_sim *bsp, realtype _p_0, realtype _p_1);

static int bloch_equations(realtype t, N_Vector M, N_Vector Mdot, void *user_data);

static int bloch_jacobian(long int N, realtype t,
        N_Vector M, N_Vector fM, DlsMat J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int bloch_root(realtype t, N_Vector y, realtype *gout, void *user_data);

static void initialize_bloch(struct bloch_sim *bsp, realtype _p_0, realtype _p_1)
{
    bsp->w_0 = 2*M_PI*60e3;
    bsp->B_0 = bsp->w_0 / GAMMA_P;
    bsp->w_1 = -0.1 * bsp->w_0;
    bsp->B_1 = -bsp->w_1 / GAMMA_P;

    bsp-> T_1 = 5e-3;
    bsp-> T_2 = 5e-3;

    bsp->p_0 = _p_0;
    bsp->p_1 = _p_1;

    bsp->num_cells = 3;
    bsp->cell_frequencies = malloc(bsp->num_cells * sizeof(realtype));
    bsp->cell_positions   = malloc(bsp->num_cells * sizeof(realtype));

    int i; realtype x; realtype w_cell; realtype w_total;
    w_total = 0;
    for (i=0; i<bsp->num_cells; ++i)
    {
        x = -1.0 + i*2.0/(bsp->num_cells - 1);
        bsp->cell_positions[i] = x;

        w_cell = bsp->w_0 + bsp->p_0*bsp->w_0*x + bsp->p_1*bsp->w_0*x*x;
        bsp->cell_frequencies[i] = w_cell;
        w_total += w_cell;
    }

    bsp->w_avg = w_total / bsp->num_cells;
    bsp->B_avg = bsp->w_avg / GAMMA_P;

    //TODO: increase pi2 samples per period to max necessary
    bsp->pi2_samples_per_period = 45;
    bsp->pi2_duration           = -M_PI / (2*bsp->w_1);
    bsp->pi2_num_samples        = bsp->pi2_samples_per_period * (int)(bsp->pi2_duration * bsp->w_avg / (2.0*M_PI));
    bsp->pi2_sampling_rate      = bsp->pi2_num_samples / bsp->pi2_duration;
    bsp->pi2_sampling_interval  = bsp->pi2_duration / bsp->pi2_num_samples;

    bsp->fid_samples_per_period = 15;
    bsp->fid_duration           = bsp->T_2 * 6;
    bsp->fid_num_samples        = bsp->fid_samples_per_period * bsp->fid_duration * bsp->w_avg / (2*M_PI);
    bsp->fid_sampling_rate      = bsp->fid_num_samples / bsp->fid_duration;
    bsp->fid_sampling_interval  = bsp->fid_duration / bsp->fid_num_samples;

    bsp->num_zero_crossings = 0;
    bsp->zero_crossings = malloc( (int)( bsp->fid_duration / (M_PI / bsp->w_avg) + 1000 ) ); // add extra 1000 just to be sure the array is long enough.

    // {{{ DEBUG BLOCH PARAMS PRINT STATEMENTS
    if (DEBUG)
    {
        printf("###################################\n");
        printf("### BLOCH SIMULATION PARAMETERS ###\n");
        printf("###################################\n");

        printf("w_0: %.5e (rad/sec), %.5e (Hz)\n", bsp->w_0, bsp->w_0 / (2*M_PI));
        printf("B_0: %.5e (Tesla)\n", bsp->B_0);
        printf("w_1: %.5e (rad/sec), %.5e (Hz)\n", bsp->w_1, bsp->w_1 / (2*M_PI));
        printf("B_1: %.5e (Tesla)\n", bsp->B_1);
        printf("T_1: %.5e (sec)\n", bsp->T_1);
        printf("T_2: %.5e (sec)\n", bsp->T_2);
        printf("p_0: %.5e (linear perturbation coefficient)\n", bsp->p_0);
        printf("p_1: %.5e (quadratic perturbation coefficient)\n", bsp->p_1);
        printf("\n");
        printf("#############\n");
        printf("### CELLS ###\n");
        printf("#############\n");
        printf("num_cells: %d\n\n", bsp->num_cells);
        int i;
        for (i=0; i<bsp->num_cells; ++i)
        {
            printf("cell %d position: %.5e\n", i, bsp->cell_positions[i]);
            printf("cell %d larmor frequency: %.5e (rad/sec), %.5e (Hz)\n", i, bsp->cell_frequencies[i], bsp->cell_frequencies[i] / (2*M_PI));
            printf("\n");
        }

        printf("w_avg: %.5e (rad/sec), %.5e (Hz)\n", bsp->w_avg, bsp->w_avg / (2*M_PI));
        printf("B_avg: %.5e (Tesla)\n", bsp->B_avg);

        printf("\n");
        printf("###############################\n");
        printf("### FID SAMPLING PROPERTIES ###\n");
        printf("###############################\n");

        printf("fid_num_samples: %d\n", bsp->fid_num_samples);
        printf("fid_samples_per_period: %d\n", bsp->fid_samples_per_period);
        printf("fid_duration: %.5e (sec)\n", bsp->fid_duration);
        printf("fid_sampling_interval: %.5e (sec)\n", bsp->fid_sampling_interval);
        printf("fid_sampling_rate: %.5e (Hz)\n", bsp->fid_sampling_rate);

        printf("\n");
        printf("######################################\n");
        printf("### PI/2 PULSE SAMPLING PROPERTIES ###\n");
        printf("######################################\n");

        printf("pi2_num_samples: %d\n", bsp->pi2_num_samples);
        printf("pi2_samples_per_period: %d\n", bsp->pi2_samples_per_period);
        printf("pi2_duration: %.5e (sec)\n", bsp->pi2_duration);
        printf("pi2_sampling_interval: %.5e (sec)\n", bsp->pi2_sampling_interval);
        printf("pi2_sampling_rate: %.5e (Hz)\n", bsp->pi2_sampling_rate);
    }
    // }}}
}

static void free_bloch(struct bloch_sim *bsp)
{
    free(bsp->cell_positions);
    free(bsp->cell_frequencies);
    free(bsp);
}

// {{{ CHECK FLAG FUNCTION
static int check_flag(void *flagvalue, char *funcname, int opt)
{
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return(1); }}

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    return(0);
}
// }}}

static int simulate_nmr_pulse(struct bloch_sim *bsp)
{
    N_Vector M = NULL;
    M = N_VNew_Serial(3 * bsp->num_cells);
    if (check_flag((void *)M, "N_VNew_Serial", 0)) return(1);

    int i;
    /* Set initial (t=0) magnetization conditions */
    for(i=0; i < bsp->num_cells; ++i)
    {
        X(M,i) = 0.0;
        Y(M,i) = 0.0;
        Z(M,i) = bsp->cell_frequencies[i] / bsp->w_avg;
    }

    realtype reltol = RCONST(1.0e-13);
    realtype abstol = RCONST(1.0e-13);

    void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    int flag;
    //TODO: check if flag should be pointer;
    flag = CVodeInit(cvode_mem, bloch_equations, 0.0, M);
    if (check_flag(&flag, "CVodeInit", 1)) return(1);

    flag = CVodeSetUserData(cvode_mem, bsp);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    flag = CVodeSStolerances(cvode_mem, reltol, abstol);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    flag = CVDense(cvode_mem, 3 * bsp->num_cells);
    if (check_flag(&flag, "CVDense", 1)) return(1);

    flag = CVDlsSetDenseJacFn(cvode_mem, bloch_jacobian);
    if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

    ///////////////////////////
    // PI/2 PULSE SIMULATION //
    ///////////////////////////
    bsp->rf_on = 1;

    flag = CVodeSetStopTime(cvode_mem, bsp->pi2_duration);
    if (check_flag(&flag, "CVodeSetStopTime", 1)) return 1;

    realtype time_reached;
    flag = CVode(cvode_mem, bsp->pi2_duration, M, &time_reached, CV_NORMAL);

    if (flag != CV_SUCCESS)
    {
        printf("ERROR: Failed to simulate Pi/2 pulse\n");
    }

    // {{{ PI2 PULSE DEBUG STATEMENTS
    if (DEBUG)
    {
        printf("\n");
        printf("#####################################\n");
        printf("### PI/2 PULSE SIMULATION DETAILS ###\n");
        printf("#####################################\n");
        printf("\n");

        printf("TIME REACHED: %.15e\n", time_reached);
        printf("TIME REACHED - PI2_DURATION: %.15e\n", time_reached - bsp->pi2_duration);

        printf("\n");
        printf("MAGNETIZATION AT END OF PI/2 PULSE:\n");
        for (i = 0; i<bsp->num_cells; ++i)
        {
            printf("CELL %d: %.4e, %.4e, %.4e\n", i, X(M,i), Y(M,i), Z(M,i));
        }
    }
    // }}}

    ////////////////////
    // FID SIMULATION //
    ////////////////////
    bsp->rf_on = 0;

    if (DEBUG)
    {
        printf("##############################\n");
        printf("### FID SIMULATION DETAILS ###\n");
        printf("##############################\n");
    }

    flag = CVodeReInit(cvode_mem, 0.0, M);
    if (check_flag(&flag, "CVodeReInit", 1)) return(1);
    time_reached = 0.0;

    flag = CVodeSetStopTime(cvode_mem, bsp->fid_duration);
    if (check_flag(&flag, "CVodeSetStopTime", 1)) return 1;

    flag = CVodeRootInit(cvode_mem, 1, bloch_root);
    if (check_flag(&flag, "CVodeRootInit", 1)) return 1;

    realtype time_desired, M_FID_LAB;
    while (time_reached < bsp->fid_duration)
    {
        if (flag == CV_SUCCESS && WRITE_FID)
        {
            M_FID_LAB = 0.0;
            for (i=0; i < bsp->num_cells; ++i)
            {
                M_FID_LAB -= X(M,i) * sin(bsp->w_avg * time_reached);
                M_FID_LAB += Y(M,i) * cos(bsp->w_avg * time_reached);
            }
            M_FID_LAB /= bsp->num_cells;

            // write fid samples to file here
        }

        time_desired = time_reached + bsp->fid_sampling_interval;
        flag = CVode(cvode_mem, time_desired, M, &time_reached, CV_NORMAL);

        if (flag == CV_ROOT_RETURN)
        {
            bsp->zero_crossings[bsp->num_zero_crossings] = time_reached;
            bsp->num_zero_crossings++;
        }

    }

    if (WRITE_FID_FREQUENCIES)
    {

    }

    N_VDestroy_Serial(M);

    CVodeFree(&cvode_mem);

    return 0;
}

static int bloch_equations(realtype t, N_Vector M, N_Vector Mdot, void *user_data)
{
    struct bloch_sim *bsp = (struct bloch_sim*)user_data;

    int i;
    realtype dw, w_1, w_cell, Mx, My, Mz;
    for (i=0; i < bsp->num_cells; ++i)
    {
        w_cell = bsp->cell_frequencies[i];
        dw =  w_cell - bsp->w_avg;
        w_1 = bsp->rf_on ? bsp->w_1 : 0.0;
        Mx = X(M,i); My = Y(M,i); Mz = Z(M,i);

        X(Mdot,i) = dw * My - Mx / bsp->T_2;
        Y(Mdot,i) = -dw * Mx + w_1 * Mz - My / bsp->T_2;
        Z(Mdot,i) = -w_1 * My + (w_cell/bsp->w_avg - Mz) / bsp->T_1;
    }

    return 0;
}

static int bloch_jacobian(long int N, realtype t,
        N_Vector M, N_Vector fM, DlsMat J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    struct bloch_sim *bsp = (struct bloch_sim*)user_data;

    int i;
    for (i=0; i < bsp->num_cells; ++i)
    {
        realtype dw = bsp->cell_frequencies[i] - bsp->w_avg;
        realtype w_1 = bsp->rf_on ? bsp->w_1 : 0.0;

        XX(J,i) = -1 / bsp->T_2;
        XY(J,i) = dw;
        XZ(J,i) = 0.0;
        YX(J,i) = -dw;
        YY(J,i) = -1 / bsp->T_2;
        YZ(J,i) = w_1;
        ZX(J,i) = 0.0;
        ZY(J,i) = -w_1;
        ZZ(J,i) = -1 / bsp->T_1;
    }

    return 0;
}

static int bloch_root(realtype t, N_Vector M, realtype *gout, void *user_data)
{
    struct bloch_sim *bsp = (struct bloch_sim*)user_data;

    gout[0] = 0.0;
    int i;
    for (i=0; i < bsp->num_cells; ++i)
    {
        gout[0] -= X(M,i) * sin(bsp->w_avg * t);
        gout[0] += Y(M,i) * cos(bsp->w_avg * t);
    }

    return 0;
}

// static void calculate_frequencies(realtype *zero_crossings, int , realtype *frequencies)
// {
// }

int
main (int argc, char **argv)
{
    struct bloch_sim *b = malloc(sizeof(*b));

    initialize_bloch(b, 0.0, 1e-3);
    simulate_nmr_pulse(b);
    free_bloch(b);
    return 0;
}
