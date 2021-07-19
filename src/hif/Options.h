/*
///////////////////////////////////////////////////////////////////////////////
//                  This file is part of the HIFIR library                   //
///////////////////////////////////////////////////////////////////////////////
*/

/*!
 * \file hif/Options.h
 * \brief HIF algorithm parameter controls
 * \author Qiao Chen

\verbatim
Copyright (C) 2021 NumGeom Group at Stony Brook University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
\endverbatim

 */

#ifndef _HIF_OPTIONS_H
#define _HIF_OPTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \addtogroup itr
 * @{
 */

/*!
 * \brief the verbose level for progress report
 */
enum {
  HIF_VERBOSE_NONE     = 0,                         /*!< mute */
  HIF_VERBOSE_INFO     = 1,                         /*!< general information */
  HIF_VERBOSE_PRE      = HIF_VERBOSE_INFO << 1,     /*!< preprocessing */
  HIF_VERBOSE_FAC      = HIF_VERBOSE_PRE << 1,      /*!< factorization update */
  HIF_VERBOSE_PRE_TIME = HIF_VERBOSE_FAC << 1,      /*! pre time */
  HIF_VERBOSE_MEM      = HIF_VERBOSE_PRE_TIME << 1, /*!< memory debug */
  HIF_VERBOSE_INFO2    = HIF_VERBOSE_MEM << 1, /*!< more detailed than info */
};

/*!
 * \brief reordering method
 */
enum {
  HIF_REORDER_OFF  = 0, /*!< turn reordering off */
  HIF_REORDER_AUTO = 1, /*!< use automaticaly reordering (default) */
  HIF_REORDER_AMD  = 2, /*!< use AMD ordering */
  HIF_REORDER_RCM  = 3, /*!< use RCM ordering (require BGL) */
  HIF_REORDER_NULL = 4, /*!< ordering Null flag (internal checking) */
};

/*!
 * \brief pivoting flag
 */
enum {
  HIF_PIVOTING_OFF  = 0, /*!< perform deferring-only factorization */
  HIF_PIVOTING_ON   = 1, /*!< perform deferred Crout factor with pivoting */
  HIF_PIVOTING_AUTO = 2, /*!< auto kernel (hybridizing between 0 and 1) */
};

/*!
 * \struct hif_Options
 * \brief POD parameter controls
 * \note Values in parentheses are default settings
 */
struct hif_Options {
  double tau_L;      /*!< inverse-based droptol for L (0.001) */
  double tau_U;      /*!< inverse-based droptol for U (0.001) */
  double kappa_d;    /*!< threshold for inverse-diagonal (3.) */
  double kappa;      /*!< inverse-norm threshold (3.) */
  double alpha_L;    /*!< growth factor of nnz per col (10) */
  double alpha_U;    /*!< growth factor of nnz per row (10) */
  double rho;        /*!< density threshold for dense LU (0.5) */
  double c_d;        /*!< size parameter for dense LU (10.0) */
  double c_h;        /*!< size parameter for H-version (2.0) */
  int    N;          /*!< reference size of matrix (-1, system size) */
  int    verbose;    /*!< message output level (1, i.e. info) */
  int    rf_par;     /*!< parameter refinement (default 1) */
  int    reorder;    /*!< reordering method (default is 2 (AMD)) */
  int    spd;        /*!< SPD-ness: 0 (ID), >0 (PD), <0 (ND), (default 0) */
  int    check;      /*!< check user input (default is true (!=0)) */
  int    pre_scale;  /*!< prescale (default 0 (off)) */
  int symm_pre_lvls; /*!< levels to be applied with symm preprocessing. If this
                        value is non-negative, then we will apply at most this
                        number of symmetric preprocessing. If this number if
                        negative, then within abs(symm_pre_lvls) levels, we will
                        automatically determine symmetric preprocessing via
                        pattern symmetry (default -2) */
  int    threads;    /*!< user specified threads (default 0) */
  int    mumps_blr;  /*!< MUMPS BLR options (default 2) *deprecated* */
  int    fat_schur_1st; /*!< double alpha for dropping L_E and U_F on 1st lvl */
  double rrqr_cond;     /*!< condition number threshold for RRQR (default 0) */
  int    pivot;         /*!< pivoting flag (default is AUTO (2)) */
  double gamma;         /*!< threshold for thresholded pivoting (1.0) */
  double beta;          /*!< safeguard factor for equlibrition scaling (1e3) */
  int    is_symm;       /*!< is symmetric (Hermitian) system? (default 0) */
  int    no_pre;        /*!< no preprocessing (default 0) */
  double nzp_thres;     /*!< nonzero pattern symmetry threshold for symm
                           preprocessing (0.65) */
  int dense_thres;      /*!< threshold for dense for Schur complement (2000) */
};

/*!
 * \typedef hif_Options
 * \brief type wrapper
 */
typedef struct hif_Options hif_Options;

/*!
 * \typedef hif_Params
 * \brief alias of \ref hif_Options
 */
typedef struct hif_Options hif_Params;

/*!
 * \brief get the default controls
 * \note See the values of attributes in parentheses
 */
static hif_Options hif_get_default_options(void) {
  return (hif_Options){.tau_L         = 0.0001,
                       .tau_U         = 0.0001,
                       .kappa_d       = 3.0,
                       .kappa         = 3.0,
                       .alpha_L       = 10.0,
                       .alpha_U       = 10.0,
                       .rho           = 0.5,
                       .c_d           = 10.0,
                       .c_h           = 2.0,
                       .N             = -1,
                       .verbose       = HIF_VERBOSE_INFO,
                       .rf_par        = 1,
                       .reorder       = HIF_REORDER_AMD,
                       .spd           = 0,
                       .check         = 1,
                       .pre_scale     = 0,
                       .symm_pre_lvls = -2,
                       .threads       = 0,
                       .mumps_blr     = 1,
                       .fat_schur_1st = 0,
                       .rrqr_cond     = 0.0,
                       .pivot         = HIF_PIVOTING_AUTO,
                       .gamma         = 1.0,
                       .beta          = 1e3,
                       .is_symm       = 0,
                       .no_pre        = 0,
                       .nzp_thres     = 0.65,
                       .dense_thres   = 2000};
}

/*!
 * \brief alias of \ref hif_get_default_options
 */
static hif_Params hif_get_default_params(void) {
  return hif_get_default_options();
}

/*!
 * \brief enable verbose flag
 * \param[in] flag verbose flag
 * \param[in,out] opts options
 */
static void hif_enable_verbose(const int flag, hif_Options *opts) {
  if (opts) {
    if (flag > 0)
      opts->verbose |= flag;
    else
      opts->verbose = HIF_VERBOSE_NONE;
  }
}

/*!
 * \brief get the string tag for reordering methods
 * \param[in] opt options
 * \return C-string of the method name
 */
static const char *hif_get_reorder_name(const hif_Options *opt) {
  if (opt) {
    switch (opt->reorder) {
      case HIF_REORDER_OFF:
        return "Off";
      case HIF_REORDER_AUTO:
        return "Auto";
      case HIF_REORDER_AMD:
        return "AMD";
      case HIF_REORDER_RCM:
        return "RCM";
      default:
        return "Null";
    }
  }
  return "Null";
}

/*!
 * @}
 */ /* c interface group */

#ifdef __cplusplus
}

/* C++ interface */
#  include <algorithm>
#  include <cmath>
#  include <functional>
#  include <string>
#  include <tuple>
#  include <unordered_map>

namespace hif {

/*!
 * \addtogroup itr
 * @{
 */

/*!
 * \brief enum wrapper
 * \note The prefix \a HIF is dropped
 */
enum : int {
  VERBOSE_NONE     = ::HIF_VERBOSE_NONE,     /*!< mute */
  VERBOSE_INFO     = ::HIF_VERBOSE_INFO,     /*!< general information */
  VERBOSE_PRE      = ::HIF_VERBOSE_PRE,      /*!< preprocessing */
  VERBOSE_FAC      = ::HIF_VERBOSE_FAC,      /*!< factorization update */
  VERBOSE_PRE_TIME = ::HIF_VERBOSE_PRE_TIME, /*!< pre time */
  VERBOSE_MEM      = ::HIF_VERBOSE_MEM,      /*!< memory debug */
  VERBOSE_INFO2    = ::HIF_VERBOSE_INFO2,    /*!< more detailed than info */
};

/*!
 * \brief enum wrapper for reordering methods
 * \note The prefix \a HIF is dropped
 */
enum : int {
  REORDER_OFF  = ::HIF_REORDER_OFF, /*!< turn reordering off */
  REORDER_AUTO = ::HIF_REORDER_AUTO,
  /*!< use automaticaly reordering (default) */
  REORDER_AMD  = ::HIF_REORDER_AMD,  /*!< use AMD ordering */
  REORDER_RCM  = ::HIF_REORDER_RCM,  /*!< use RCM ordering (require BGL) */
  REORDER_NULL = ::HIF_REORDER_NULL, /*!< ordering Null flag */
};

/*!
 * \brief enum wrapper for pivoting
 * \note The prefix \a HIF is dropped
 */
enum : int {
  PIVOTING_OFF  = ::HIF_PIVOTING_OFF,
  PIVOTING_ON   = ::HIF_PIVOTING_ON,
  PIVOTING_AUTO = ::HIF_PIVOTING_AUTO,
};

/*!
 * \typedef Options
 * \brief type wrapper
 */
typedef hif_Options Options;

/*!
 * \typedef Params
 * \brief alias of \ref Options
 */
typedef Options Params;

/*!
 * \brief get the reordering method name
 */
inline std::string get_reorder_name(const Options &opt) {
  return ::hif_get_reorder_name(&opt);
}

/*!
 * \brief enable verbose flags
 * \param[in] flag verbose options
 * \param[in,out] opts options
 */
inline void enable_verbose(const int flag, Options &opts) {
  ::hif_enable_verbose(flag, &opts);
}

/*!
 * \brief get the verbose name
 */
inline std::string get_verbose(const Options &opt);

/*!
 * \brief get the default configuration
 */
inline Options get_default_options() { return ::hif_get_default_options(); }

/*!
 * \brief alias of \ref get_default_options
 */
inline Params get_default_params() { return ::hif_get_default_params(); }

/*!
 * \var DEFAULT_PARAMS
 * \brief global default parameters
 */
const static Params DEFAULT_PARAMS = get_default_params();

/*!
 * \brief represent an option control with C++ string
 * \param[in] opt input option controls
 * \return string representation of \a opt
 */
inline std::string opt_repr(const Options &opt) {
  using std::string;
  using std::to_string;               /* C++11 */
  const static int leading_size = 30; /* should be enough */
  const auto       pack_int     = [](const string &cat, const int v) -> string {
    return cat + string(leading_size - cat.size(), ' ') + to_string(v) + "\n";
  };
  const auto pack_double = [](const string &cat, const double v) -> string {
    return cat + string(leading_size - cat.size(), ' ') + to_string(v) + "\n";
  };
  const auto pack_name =
      [&](const string &                                cat,
          const std::function<string(const Options &)> &dec) -> string {
    return cat + string(leading_size - cat.size(), ' ') + dec(opt) + "\n";
  };
  return pack_double("tau_L", opt.tau_L) + pack_double("tau_U", opt.tau_U) +
         pack_double("kappa_d", opt.kappa_d) + pack_double("kappa", opt.kappa) +
         pack_double("alpha_L", opt.alpha_L) +
         pack_double("alpha_U", opt.alpha_U) + pack_double("rho", opt.rho) +
         pack_double("c_d", opt.c_d) + pack_double("c_h", opt.c_h) +
         pack_int("N", opt.N) + pack_name("verbose", get_verbose) +
         pack_int("rf_par", opt.rf_par) +
         pack_name("reorder", get_reorder_name) + pack_int("spd", opt.spd) +
         pack_name(
             "check",
             [](const Options &opt_) { return opt_.check ? "yes" : "no"; }) +
         pack_int("pre_scale", opt.pre_scale) +
         pack_int("symm_pre_lvls", opt.symm_pre_lvls) +
         pack_int("threads", opt.threads) +
         pack_int("fat_schur_1st", opt.fat_schur_1st) +
         pack_double("rrqr_cond", opt.rrqr_cond) +
         pack_name("pivot",
                   [](const Options &opt_) {
                     return opt_.pivot == PIVOTING_OFF
                                ? "off"
                                : (opt_.pivot == PIVOTING_ON ? "on" : "auto");
                   }) +
         pack_double("gamma", opt.gamma) + pack_double("beta", opt.beta) +
         pack_int("is_symm", opt.is_symm) + pack_int("no_pre", opt.no_pre) +
         pack_double("nzp_thres", opt.nzp_thres) +
         pack_int("dense_thres", opt.dense_thres);
}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
#    define _HIF_TOTAL_OPTIONS 28

/* data type tags, true for double, false for int */
const static bool option_dtypes[_HIF_TOTAL_OPTIONS] = {
    true,   // 0
    true,   // 1
    true,   // 2
    true,   // 3
    true,   // 4
    true,   // 5
    true,   // 6
    true,   // 7
    true,   // 8
    false,  // 9
    false,  // 10
    false,  // 11
    false,  // 12
    false,  // 13
    false,  // 14
    false,  // 15
    false,  // 16
    false,  // 17
    false,  // 18
    false,  // 19
    true,   // 20
    false,  // 21
    true,   // 22
    true,   // 23
    false,  // 24
    false,  // 25
    true,   // 26
    false,  // 27
};

/* using unordered map to store the string to index map */
const static std::unordered_map<std::string, int> option_tag2pos = {
    {"tau_L", 0},
    {"tau_U", 1},
    {"kappa_d", 2},
    {"kappa", 3},
    {"alpha_L", 4},
    {"alpha_U", 5},
    {"rho", 6},
    {"c_d", 7},
    {"c_h", 8},
    {"N", 9},
    {"verbose", 10},
    {"rf_par", 11},
    {"reorder", 12},
    {"spd", 13},
    {"check", 14},
    {"pre_scale", 15},
    {"symm_pre_lvls", 16},
    {"threads", 17},
    {"mumps_blr", 18},
    {"fat_schur_1st", 19},
    {"rrqr_cond", 20},
    {"pivot", 21},
    {"gamma", 22},
    {"beta", 23},
    {"is_symm", 24},
    {"no_pre", 25},
    {"nzp_thres", 26},
    {"dense_thres", 27}};

} /* namespace internal */
#  endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// \brief set \ref Options attribute value from key value pairs
/// \tparam T value type, either \a double or \a int
/// \param[in] attr attribute/member name
/// \param[in] v value
/// \param[out] opt output options
///
/// This function can be handy while initialing option parameters from string
/// values. Notice that the keys (string values) are the same as the attribute
/// variable names.
template <typename T>
inline bool set_option_attr(const std::string &attr, const T v, Options &opt) {
  const static bool failed = true;
  int               pos;
  try {
    pos = internal::option_tag2pos.at(attr);
  } catch (const std::out_of_range &) {
    return failed;
  }
  switch (pos) {
    case 0:
      opt.tau_L = v;
      break;
    case 1:
      opt.tau_U = v;
      break;
    case 2:
      opt.kappa_d = v;
      break;
    case 3:
      opt.kappa = v;
      break;
    case 4:
      opt.alpha_L = v;
      break;
    case 5:
      opt.alpha_U = v;
      break;
    case 6:
      opt.rho = v;
      break;
    case 7:
      opt.c_d = v;
      break;
    case 8:
      opt.c_h = v;
      break;
    case 9:
      opt.N = v;
      break;
    case 10:
      opt.verbose = v;
      break;
    case 11:
      opt.rf_par = v;
      break;
    case 12:
      opt.reorder = v;
      break;
    case 13:
      opt.spd = v;
      break;
    case 14:
      opt.check = v;
      break;
    case 15:
      opt.pre_scale = v;
      break;
    case 16:
      opt.symm_pre_lvls = v;
      break;
    case 17:
      opt.threads = v;
      break;
    case 18:
      opt.mumps_blr = v;
      break;
    case 19:
      opt.fat_schur_1st = v;
      break;
    case 20:
      opt.rrqr_cond = v;
      break;
    case 21:
      opt.pivot = v;
      break;
    case 22:
      opt.gamma = v;
      break;
    case 23:
      opt.beta = v;
      break;
    case 24:
      opt.is_symm = v;
      break;
    case 25:
      opt.no_pre = v;
      break;
    case 26:
      opt.nzp_thres = v;
      break;
    case 27:
      opt.dense_thres = v;
  }
  return !failed;
}

/// \brief Set the param attributes/values
/// \note Alias of \ref set_option_attr
template <typename T>
inline bool set_param_attr(const std::string &attr, const T v, Options &opt) {
  return set_option_attr(attr, v, opt);
}

/*!
 * @}
 */

} /* namespace hif */

/*!
 * \brief read control parameters from a standard input streamer
 * \tparam InStream input streamer, i.e. with input operator
 * \param[in,out] in_str input streamer, e.g. \a std::cin
 * \param[out] opt control parameters
 * \return reference to \a in_str to enable chain reaction
 * \note Read data in sequential order with default separators
 * \ingroup itr
 */
template <class InStream>
inline InStream &operator>>(InStream &in_str, hif::Options &opt) {
  in_str >> opt.tau_L >> opt.tau_U >> opt.kappa_d >> opt.kappa >> opt.alpha_L >>
      opt.alpha_U >> opt.rho >> opt.c_d >> opt.c_h >> opt.N >> opt.verbose >>
      opt.rf_par >> opt.reorder >> opt.spd >> opt.check >> opt.pre_scale >>
      opt.symm_pre_lvls >> opt.threads >> opt.mumps_blr >> opt.fat_schur_1st >>
      opt.rrqr_cond >> opt.gamma >> opt.beta >> opt.is_symm >> opt.no_pre >>
      opt.nzp_thres >> opt.dense_thres;
  return in_str;
}

/*!
 * \def hif_verbose2(__LVL, __opt_tag)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 */
#  define hif_verbose2(__LVL, __opt_tag) (__opt_tag & ::hif::VERBOSE_##__LVL)

/*!
 * \def hif_verbose(__LVL, __opt)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 *
 * \code{.cpp}
 * if (hif_verbose(INFO, opt)) ...;
 * \endcode
 */
#  define hif_verbose(__LVL, __opt) hif_verbose2(__LVL, __opt.verbose)

inline std::string hif::get_verbose(const hif::Options &opt) {
  std::string name("");
  if (opt.verbose == VERBOSE_NONE)
    name = "none";
  else {
    if (hif_verbose(INFO, opt)) name = "info";
    if (hif_verbose(PRE, opt)) {
      if (name != "")
        name += ",pre";
      else
        name = "pre";
    }
    if (hif_verbose(FAC, opt)) {
      if (name != "")
        name += ",fac";
      else
        name = "fac";
    }
    if (hif_verbose(PRE_TIME, opt)) {
      if (name != "")
        name += ",pre_time";
      else
        name = "pre_time";
    }
    if (hif_verbose(MEM, opt)) {
      if (name != "")
        name += ",mem";
      else
        name = "mem";
    }
  }
  return name;
}

#endif /* __cplusplus */

#endif /* _HIF_OPTIONS_H */
