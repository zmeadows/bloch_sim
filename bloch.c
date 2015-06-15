#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

#include <assert.h>
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

    realtype pi2_duration;

    int estimated_num_zero_crossings;
    int num_zero_crossings;
    realtype *zero_crossings;
    realtype *envelope;

};

static void initialize_bloch(struct bloch_sim *bs, realtype _p_0, realtype _p_1);

static int bloch_equations(realtype t, N_Vector M, N_Vector Mdot, void *user_data);

static int bloch_jacobian(long int N, realtype t,
        N_Vector M, N_Vector fM, DlsMat J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int bloch_root(realtype t, N_Vector y, realtype *gout, void *user_data);

// static void print_diagnostics(struct bloch_sim *bs);
// static void print_frequencies(struct bloch_sim *bs);

static void initialize_bloch(struct bloch_sim *bs, realtype _p_0, realtype _p_1)
{
    bs->w_0 = 2*M_PI*60e6;
    bs->B_0 = bs->w_0 / GAMMA_P;
    bs->w_1 = -0.1 * bs->w_0;
    bs->B_1 = -bs->w_1 / GAMMA_P;

    bs-> T_1 = 5e-3;
    bs-> T_2 = 5e-3;

    bs->p_0 = _p_0;
    bs->p_1 = _p_1;

    bs->num_cells = 3;
    bs->cell_frequencies = malloc(bs->num_cells * sizeof(realtype));
    bs->cell_positions   = malloc(bs->num_cells * sizeof(realtype));

    int i; realtype x; realtype w_cell; realtype w_total;
    w_total = 0;
    for (i=0; i<bs->num_cells; ++i)
    {
        x = -1.0 + i*2.0/(bs->num_cells - 1);
        bs->cell_positions[i] = x;

        w_cell = bs->w_0 + bs->p_0*bs->w_0*x + bs->p_1*bs->w_0*x*x;
        bs->cell_frequencies[i] = w_cell;
        w_total += w_cell;
    }

    bs->w_avg = w_total / bs->num_cells;
    bs->B_avg = bs->w_avg / GAMMA_P;

    bs->pi2_duration           = -M_PI / (2*bs->w_1);

    bs->fid_samples_per_period = 45;
    bs->fid_duration           = bs->T_2 * 6;
    bs->fid_num_samples        = bs->fid_samples_per_period * bs->fid_duration * bs->w_avg / (2*M_PI);
    bs->fid_sampling_rate      = bs->fid_num_samples / bs->fid_duration;
    bs->fid_sampling_interval  = bs->fid_duration / bs->fid_num_samples;

    bs->num_zero_crossings = 0;
    bs->estimated_num_zero_crossings = (int)( bs->fid_duration / (M_PI / bs->w_avg));
    bs->zero_crossings = malloc( sizeof(realtype) * bs->estimated_num_zero_crossings * 2 );
    bs->envelope = malloc( sizeof(realtype) * bs->estimated_num_zero_crossings * 2 );

    // {{{ DEBUG BLOCH PARAMS PRINT STATEMENTS
    if (DEBUG)
    {
        printf("###################################\n");
        printf("### BLOCH SIMULATION PARAMETERS ###\n");
        printf("###################################\n");

        printf("w_0: %.5e (rad/sec), %.5e (Hz)\n", bs->w_0, bs->w_0 / (2*M_PI));
        printf("B_0: %.5e (Tesla)\n", bs->B_0);
        printf("w_1: %.5e (rad/sec), %.5e (Hz)\n", bs->w_1, bs->w_1 / (2*M_PI));
        printf("B_1: %.5e (Tesla)\n", bs->B_1);
        printf("T_1: %.5e (sec)\n", bs->T_1);
        printf("T_2: %.5e (sec)\n", bs->T_2);
        printf("p_0: %.5e (linear perturbation coefficient)\n", bs->p_0);
        printf("p_1: %.5e (quadratic perturbation coefficient)\n", bs->p_1);
        printf("\n");
        printf("#############\n");
        printf("### CELLS ###\n");
        printf("#############\n");
        printf("num_cells: %d\n\n", bs->num_cells);
        int i;
        for (i=0; i<bs->num_cells; ++i)
        {
            printf("cell %d position: %.5e\n", i, bs->cell_positions[i]);
            printf("cell %d larmor frequency: %.5e (rad/sec), %.5e (Hz)\n", i, bs->cell_frequencies[i], bs->cell_frequencies[i] / (2*M_PI));
            printf("\n");
        }

        printf("w_avg: %.5e (rad/sec), %.5e (Hz)\n", bs->w_avg, bs->w_avg / (2*M_PI));
        printf("B_avg: %.5e (Tesla)\n", bs->B_avg);

        printf("\n");
        printf("###############################\n");
        printf("### FID SAMPLING PROPERTIES ###\n");
        printf("###############################\n");

        printf("fid_num_samples: %d\n", bs->fid_num_samples);
        printf("fid_samples_per_period: %d\n", bs->fid_samples_per_period);
        printf("fid_duration: %.5e (sec)\n", bs->fid_duration);
        printf("fid_sampling_interval: %.5e (sec)\n", bs->fid_sampling_interval);
        printf("fid_sampling_rate: %.5e (Hz)\n", bs->fid_sampling_rate);

        printf("\n");
        printf("######################################\n");
        printf("### PI/2 PULSE SAMPLING PROPERTIES ###\n");
        printf("######################################\n");

        printf("pi2_duration: %.5e (sec)\n", bs->pi2_duration);
    }
    // }}}
}

static void free_bloch(struct bloch_sim *bs)
{
    free(bs->cell_positions);
    free(bs->cell_frequencies);
    free(bs->zero_crossings);
    free(bs->envelope);
    free(bs);
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

static int simulate_nmr_pulse(struct bloch_sim *bs)
{
    N_Vector M = NULL;
    M = N_VNew_Serial(3 * bs->num_cells);
    if (check_flag((void *)M, "N_VNew_Serial", 0)) return(1);

    int i;
    /* Set initial (t=0) magnetization conditions */
    for(i=0; i < bs->num_cells; ++i)
    {
        X(M,i) = 0.0;
        Y(M,i) = 0.0;
        Z(M,i) = bs->cell_frequencies[i] / bs->w_avg;
    }

    realtype reltol = RCONST(1.0e-14);
    realtype abstol = RCONST(1.0e-14);

    void *cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    int flag;
    //TODO: check if flag should be pointer;
    flag = CVodeInit(cvode_mem, bloch_equations, 0.0, M);
    if (check_flag(&flag, "CVodeInit", 1)) return(1);

    flag = CVodeSetUserData(cvode_mem, bs);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    flag = CVodeSStolerances(cvode_mem, reltol, abstol);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    flag = CVDense(cvode_mem, 3 * bs->num_cells);
    if (check_flag(&flag, "CVDense", 1)) return(1);

    flag = CVDlsSetDenseJacFn(cvode_mem, bloch_jacobian);
    if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

    ///////////////////////////
    // PI/2 PULSE SIMULATION //
    ///////////////////////////
    bs->rf_on = 1;

    flag = CVodeSetStopTime(cvode_mem, bs->pi2_duration);
    if (check_flag(&flag, "CVodeSetStopTime", 1)) return 1;

    realtype time_reached;
    flag = CVode(cvode_mem, bs->pi2_duration, M, &time_reached, CV_NORMAL);

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
        printf("TIME REACHED - PI2_DURATION: %.15e\n", time_reached - bs->pi2_duration);

        printf("\n");
        printf("MAGNETIZATION AT END OF PI/2 PULSE:\n");
        for (i = 0; i<bs->num_cells; ++i)
        {
            printf("CELL %d: %.4e, %.4e, %.4e\n", i, X(M,i), Y(M,i), Z(M,i));
        }
    }
    // }}}

    ////////////////////
    // FID SIMULATION //
    ////////////////////
    bs->rf_on = 0;

    if (DEBUG)
    {
        printf("##############################\n");
        printf("### FID SIMULATION DETAILS ###\n");
        printf("##############################\n");
    }

    flag = CVodeReInit(cvode_mem, 0.0, M);
    if (check_flag(&flag, "CVodeReInit", 1)) return(1);
    time_reached = 0.0;

    flag = CVodeSetStopTime(cvode_mem, bs->fid_duration);
    if (check_flag(&flag, "CVodeSetStopTime", 1)) return 1;

    flag = CVodeRootInit(cvode_mem, 1, bloch_root);
    if (check_flag(&flag, "CVodeRootInit", 1)) return 1;

    realtype time_desired, M_FID_X;
    while (time_reached < bs->fid_duration)
    {
        time_desired = time_reached + bs->fid_sampling_interval;
        flag = CVode(cvode_mem, time_desired, M, &time_reached, CV_NORMAL);

        if (flag == CV_ROOT_RETURN)
        {
            bs->zero_crossings[bs->num_zero_crossings] = time_reached;

            M_FID_X = 0.0;
            for (i=0; i < bs->num_cells; i++)
            {
                M_FID_X += X(M,i) * cos(bs->w_avg * time_reached);
                M_FID_X += Y(M,i) * sin(bs->w_avg * time_reached);
            }
            bs->envelope[bs->num_zero_crossings] = M_FID_X;
            bs->num_zero_crossings++;
        }
    }

    bs->zero_crossings = (realtype*) realloc(bs->zero_crossings, sizeof(realtype) * bs->num_zero_crossings);
    bs->envelope = (realtype*) realloc(bs->envelope, sizeof(realtype) * bs->num_zero_crossings);
    if (!bs->zero_crossings || !bs->envelope) {
        printf("ERROR: reallocating zero crossing and/or envelope array memory failed!\n");
        exit(1);
    }

    N_VDestroy_Serial(M);

    CVodeFree(&cvode_mem);

    return 0;
}

static int bloch_equations(realtype t, N_Vector M, N_Vector Mdot, void *user_data)
{
    struct bloch_sim *bs = (struct bloch_sim*)user_data;

    int i;
    realtype dw, w_1, w_cell, Mx, My, Mz;
    for (i=0; i < bs->num_cells; ++i)
    {
        w_cell = bs->cell_frequencies[i];
        dw =  w_cell - bs->w_avg;
        w_1 = bs->rf_on ? bs->w_1 : 0.0;
        Mx = X(M,i); My = Y(M,i); Mz = Z(M,i);

        X(Mdot,i) = dw * My - Mx / bs->T_2;
        Y(Mdot,i) = -dw * Mx + w_1 * Mz - My / bs->T_2;
        Z(Mdot,i) = -w_1 * My + (w_cell/bs->w_avg - Mz) / bs->T_1;
    }

    return 0;
}

static int bloch_jacobian(long int N, realtype t,
        N_Vector M, N_Vector fM, DlsMat J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    struct bloch_sim *bs = (struct bloch_sim*)user_data;

    int i;
    for (i=0; i < bs->num_cells; ++i)
    {
        realtype dw = bs->cell_frequencies[i] - bs->w_avg;
        realtype w_1 = bs->rf_on ? bs->w_1 : 0.0;

        XX(J,i) = -1 / bs->T_2;
        XY(J,i) = dw;
        XZ(J,i) = 0.0;
        YX(J,i) = -dw;
        YY(J,i) = -1 / bs->T_2;
        YZ(J,i) = w_1;
        ZX(J,i) = 0.0;
        ZY(J,i) = -w_1;
        ZZ(J,i) = -1 / bs->T_1;
    }

    return 0;
}

static int bloch_root(realtype t, N_Vector M, realtype *gout, void *user_data)
{
    struct bloch_sim *bs = (struct bloch_sim*)user_data;

    gout[0] = 0.0;
    int i;
    for (i=0; i < bs->num_cells; ++i)
    {
        gout[0] -= X(M,i) * sin(bs->w_avg * t);
        gout[0] += Y(M,i) * cos(bs->w_avg * t);
    }

    return 0;
}

void
write_frequencies(struct bloch_sim *bs, FILE *file)
{
    const int N = 1000;
    assert(bs->num_zero_crossings > 0);

    int i; realtype t, t1, t2, freq;
    t = 0.0;
    freq = 0.0;
    for (i = 1; i < bs->num_zero_crossings; i++)
    {
        t1 = bs->zero_crossings[i-1];
        t2 = bs->zero_crossings[i];
        t += 0.5 * (t1 + t2);
        freq += 0.5 / (t2 - t1);
        if (i % N == 0)
        {
            fprintf(file, "%.12e %.12e\n", t / N, freq / N - bs->w_avg/(2*M_PI));
            t = 0.0;
            freq = 0.0;
        }
    }

    fclose(file);
}


void
write_envelope(struct bloch_sim *bs, FILE *file)
{
    assert(bs->num_zero_crossings > 0);

    int i;
    for (i = 0; i < bs->num_zero_crossings; i++)
    {
        if (i % 23 == 0)
        {
            fprintf(file, "%.12e %.12e\n", bs->zero_crossings[i], bs->envelope[i]);
        }
    }

    fclose(file);
}

int
main (int argc, char **argv)
{
    struct bloch_sim *b = malloc(sizeof(*b));

    int option = 0;

    realtype p_0 = 0.0;
    realtype p_1 = 0.0;

    FILE *frequencies_file;
    FILE *envelope_file;
    // FILE *params_file;

    while ((option = getopt(argc, argv,"l:q:h:e:p:")) != -1)
    {
        switch (option)
        {
            case 'l' :
                p_0 = strtod(optarg, NULL);
                break;

            case 'q' :
                p_1 = strtod(optarg, NULL);
                break;

            case 'h' :
                frequencies_file = fopen(optarg, "w");
                if (frequencies_file == NULL) {
                    fprintf(stderr, "ERROR: failed to open output file!\n");
                    exit(1);
                }
                break;

            case 'e' :
                envelope_file = fopen(optarg, "w");
                if (envelope_file == NULL) {
                    fprintf(stderr, "ERROR: failed to open output file!\n");
                    exit(1);
                }
                break;

            default:
                //print_usage();
                break;
        }
    }


    initialize_bloch(b, p_0, p_1);
    simulate_nmr_pulse(b);

    if (frequencies_file != NULL) { write_frequencies(b, frequencies_file); }
    if (envelope_file != NULL) { write_envelope(b, envelope_file); }

    free_bloch(b);
    return 0;
}
