/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The PSMILU AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file psmilu_Options.h
 * \brief PS-MILU algorithm parameter controls
 * \authors Qiao,
 * \note Compatible with C99, and must be \b C99 or higher!
 */

#ifndef _PSMILU_OPTIONS_H
#define _PSMILU_OPTIONS_H

/*! \addtogroup itr
 *@{
 */

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief the verbose level for progress report
 */
enum {
  PSMILU_VERBOSE_NONE  = 0,                         /*!< mute */
  PSMILU_VERBOSE_INFO  = 1,                         /*!< general information */
  PSMILU_VERBOSE_PRE   = PSMILU_VERBOSE_INFO << 1,  /*!< preprocessing */
  PSMILU_VERBOSE_CROUT = PSMILU_VERBOSE_PRE << 1,   /*!< Crout update */
  PSMILU_VERBOSE_PIVOT = PSMILU_VERBOSE_CROUT << 1, /*!< pivoting */
  PSMILU_VERBOSE_THRES = PSMILU_VERBOSE_PIVOT << 1, /*!< inverse-based thres */
  PSMILU_VERBOSE_SCHUR = PSMILU_VERBOSE_THRES << 1, /*!< computing Schur */
  PSMILU_VERBOSE_MEM   = PSMILU_VERBOSE_SCHUR << 1, /*!< memory debug */
};

/*!
 * \struct psmilu_Options
 * \brief POD parameter controls
 * \note Values in parentheses are default settings
 */
struct psmilu_Options {
  double tau_L;     /*!< inverse-based threshold for L (0.01) */
  double tau_U;     /*!< inverse-based threshold for U (0.01) */
  double tau_d;     /*!< threshold for inverse-diagonal (10.) */
  double tau_kappa; /*!< inverse-norm threshold (100.) */
  int    alpha_L;   /*!< growth factor of nnz per col (4) */
  int    alpha_U;   /*!< growth factor of nnz per row (4) */
  double rho;       /*!< density threshold for dense LU (0.25) */
  int    c_d;       /*!< size parameter for dense LU (1) */
  int    c_h;       /*!< size parameter for H-version (10) */
  int    N;         /*!< reference size of matrix (-1, system size) */
  int    verbose;   /*!< message output level (1, i.e. info) */
};

/*!
 * \typedef psmilu_Options
 * \brief type wrapper
 */
typedef struct psmilu_Options psmilu_Options;

/*!
 * \brief get the default controls
 * \note See the values of attributes in parentheses
 */
inline psmilu_Options psmilu_get_default_options(void) {
  return (psmilu_Options){.tau_L     = 0.01,
                          .tau_U     = 0.01,
                          .tau_d     = 10.0,
                          .tau_kappa = 100.0,
                          .alpha_L   = 4,
                          .alpha_U   = 4,
                          .rho       = 0.25,
                          .c_d       = 1,
                          .c_h       = 10,
                          .N         = -1,
                          .verbose   = PSMILU_VERBOSE_INFO};
}

#ifdef __cplusplus
}

/* C++ interface */
#  include <string>

namespace psmilu {

#  ifndef DOXYGEN_SHOULD_SKIP_THIS
#    define ALIGN_LVL(__LVL) VERBOSE_##__LVL = ::PSMILU_VERBOSE_##__LVL
#  endif /* DOXYGEN_SHOULD_SKIP_THIS */

/*!
 * \brief enum wrapper
 * \note The prefix of \a PSMILU will be dropped
 */
enum : int {
  ALIGN_LVL(NONE),
  ALIGN_LVL(INFO),
  ALIGN_LVL(PRE),
  ALIGN_LVL(CROUT),
  ALIGN_LVL(PIVOT),
  ALIGN_LVL(THRES),
  ALIGN_LVL(SCHUR),
  ALIGN_LVL(MEM),
};

#  ifdef ALIGN_LVL
#    undef ALIGN_LVL
#  endif /* ALIGN_LVL */

/*!
 * \typedef Options
 * \brief type wrapper
 */
typedef psmilu_Options Options;

/*!
 * \brief get the default configuration
 */
inline Options get_default_options() { return ::psmilu_get_default_options(); }

/*!
 * \brief represent an option control with C++ string
 * \param[in] opt input option controls
 * \return string representation of \a opt
 */
inline std::string opt_repr(const Options &opt) {
  using std::string;
  using std::to_string;                /* C++11 */
  const static int leading_size = 30;  /* should be enough */
  const auto       pack_int     = [](const string &cat, const int v) -> string {
    return cat + string(leading_size - cat.size(), ' ') + to_string(v) + "\n";
  };
  const auto pack_double = [](const string &cat, const double v) -> string {
    return cat + string(leading_size - cat.size(), ' ') + to_string(v) + "\n";
  };
  return pack_double("tau_L", opt.tau_L) +
         pack_double("tau_U", opt.tau_U) +
         pack_double("tau_d", opt.tau_d) +
         pack_double("tau_kappa", opt.tau_kappa) +
         pack_int("alpha_L", opt.alpha_L) +
         pack_int("alpha_U", opt.alpha_U) +
         pack_double("rho", opt.rho) +
         pack_int("c_d", opt.c_d) +
         pack_int("N", opt.N) +
         pack_int("verbose", opt.verbose);
}

}  // namespace psmilu

/*!
 * \def psmilu_verbose(__LVL, __opt)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 *
 * \code{.cpp}
 * if (psmilu_verbose(INFO, opt)) ...;
 * \endcode
 */
#  define psmilu_verbose(__LVL, __opt) \
    (__opt.verbose & ::psmilu::VERBOSE_##__LVL)

#endif /* __cplusplus */

/*! @}*/  /* interface group */

#endif /* _PSMILU_OPTIONS_H */
