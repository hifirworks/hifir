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

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \addtogroup c
 * @{
 */

/*!
 * \brief the verbose level for progress report
 */
enum {
  PSMILU_VERBOSE_NONE = 0,                        /*!< mute */
  PSMILU_VERBOSE_INFO = 1,                        /*!< general information */
  PSMILU_VERBOSE_PRE  = PSMILU_VERBOSE_INFO << 1, /*!< preprocessing */
  PSMILU_VERBOSE_FAC  = PSMILU_VERBOSE_PRE << 1,  /*!< factorization update */
  PSMILU_VERBOSE_MEM  = PSMILU_VERBOSE_FAC << 1,  /*!< memory debug */
};

/*!
 * \brief reordering method
 */
enum {
  PSMILU_REORDER_OFF   = 0, /*!< turn reordering off */
  PSMILU_REORDER_AUTO  = 1, /*!< use automaticaly reordering (default) */
  PSMILU_REORDER_AMD   = 2, /*!< use AMD ordering */
  PSMILU_REORDER_RCM   = 3, /*!< use RCM ordering (require BGL) */
  PSMILU_REORDER_KING  = 4, /*!< use King ordering (require BGL) */
  PSMILU_REORDER_SLOAN = 5, /*!< use Sloan ordering (require BGL) */
  PSMILU_REORDER_NULL  = 6, /*!< ordering Null flag */
};

/*!
 * \struct psmilu_Options
 * \brief POD parameter controls
 * \note Values in parentheses are default settings
 */
struct psmilu_Options {
  double tau_L;     /*!< inverse-based threshold for L (0.001) */
  double tau_U;     /*!< inverse-based threshold for U (0.001) */
  double tau_d;     /*!< threshold for inverse-diagonal (3.) */
  double tau_kappa; /*!< inverse-norm threshold (3.) */
  int    alpha_L;   /*!< growth factor of nnz per col (8) */
  int    alpha_U;   /*!< growth factor of nnz per row (8) */
  double rho;       /*!< density threshold for dense LU (0.5) */
  double c_d;       /*!< size parameter for dense LU (10.0) */
  double c_h;       /*!< size parameter for H-version (2.0) */
  int    N;         /*!< reference size of matrix (-1, system size) */
  int    verbose;   /*!< message output level (1, i.e. info) */
  int    rf_par;    /*!< parameter refinement (default 1) */
  int    reorder;   /*!< reordering method */
  int    saddle;    /*!< enable saddle point static deferring (default 1) */
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
static psmilu_Options psmilu_get_default_options(void) {
  return (psmilu_Options){.tau_L     = 0.001,
                          .tau_U     = 0.001,
                          .tau_d     = 3.0,
                          .tau_kappa = 3.0,
                          .alpha_L   = 8,
                          .alpha_U   = 8,
                          .rho       = 0.5,
                          .c_d       = 10.0,
                          .c_h       = 2.0,
                          .N         = -1,
                          .verbose   = PSMILU_VERBOSE_INFO,
                          .rf_par    = 1,
                          .reorder   = PSMILU_REORDER_AUTO,
                          .saddle    = 1};
}

/*!
 * \brief get the string tag for reordering methods
 * \return C-string of the method name
 */
static const char *psmilu_get_reorder_name(const psmilu_Options *opt) {
  if (opt) {
    switch (opt->reorder) {
      case PSMILU_REORDER_OFF:
        return "Off";
      case PSMILU_REORDER_AUTO:
        return "Auto";
      case PSMILU_REORDER_AMD:
        return "AMD";
      case PSMILU_REORDER_RCM:
        return "RCM";
      case PSMILU_REORDER_KING:
        return "King";
      case PSMILU_REORDER_SLOAN:
        return "Sloan";
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

namespace psmilu {

/*!
 * \brief enum wrapper
 * \note The prefix of \a PSMILU will be dropped
 * \ingroup cpp
 */
enum : int {
  VERBOSE_NONE = ::PSMILU_VERBOSE_NONE, /*!< mute */
  VERBOSE_INFO = ::PSMILU_VERBOSE_INFO, /*!< general information */
  VERBOSE_PRE  = ::PSMILU_VERBOSE_PRE,  /*!< preprocessing */
  VERBOSE_FAC  = ::PSMILU_VERBOSE_FAC,  /*!< factorization update */
  VERBOSE_MEM  = ::PSMILU_VERBOSE_MEM,  /*!< memory debug */
};

/*!
 * \brief enum wrapper for reordering methods
 * \note The prefix of \a PSMILU will be dropped
 * \ingroup cpp
 */
enum : int {
  REORDER_OFF  = ::PSMILU_REORDER_OFF, /*!< turn reordering off */
  REORDER_AUTO = ::PSMILU_REORDER_AUTO,
  /*!< use automaticaly reordering (default) */
  REORDER_AMD   = ::PSMILU_REORDER_AMD,  /*!< use AMD ordering */
  REORDER_RCM   = ::PSMILU_REORDER_RCM,  /*!< use RCM ordering (require BGL) */
  REORDER_KING  = ::PSMILU_REORDER_KING, /*!< use King ordering (require BGL) */
  REORDER_SLOAN = ::PSMILU_REORDER_SLOAN,
  /*!< use Sloan ordering (require BGL) */
  REORDER_NULL = ::PSMILU_REORDER_NULL, /*!< ordering Null flag */
};

/*!
 * \typedef Options
 * \brief type wrapper
 * \ingroup cpp
 */
typedef psmilu_Options Options;

/*!
 * \brief get the reordering method name
 */
inline std::string get_reorder_name(const Options &opt) {
  return ::psmilu_get_reorder_name(&opt);
}

/*!
 * \brief get the verbose name
 */
inline std::string get_verbose(const Options &opt);

/*!
 * \brief adjust parameters based on levels
 * \param[in] opts control parameters, i.e. Options
 * \param[in] lvl levels
 * \ingroup fac
 */
inline std::tuple<double, double, double, double, int, int> determine_fac_pars(
    const Options &opts, const int lvl) {
  double tau_d, tau_kappa, tau_U, tau_L;
  int    alpha_L, alpha_U;
  if (opts.rf_par) {
    const int    fac  = std::min(lvl, 2);
    const double fac2 = 1. / std::min(10.0, std::pow(10.0, lvl - 1));
    tau_d             = std::max(2.0, std::pow(opts.tau_d, 1. / fac));
    tau_kappa         = std::max(2.0, std::pow(opts.tau_kappa, 1. / fac));
    tau_U             = opts.tau_U * fac2;
    tau_L             = opts.tau_L * fac2;
    if (lvl > 2) {
      alpha_L = opts.alpha_L;
      alpha_U = opts.alpha_U;
    } else {
      alpha_L = opts.alpha_L * fac;
      alpha_U = opts.alpha_U * fac;
    }
  } else {
    tau_d     = opts.tau_d;
    tau_kappa = opts.tau_kappa;
    tau_U     = opts.tau_U;
    tau_L     = opts.tau_L;
    alpha_L   = opts.alpha_L;
    alpha_U   = opts.alpha_U;
  }
  return std::make_tuple(tau_d, tau_kappa, tau_U, tau_L, alpha_L, alpha_U);
}

/*!
 * \brief read control parameters from a standard input streamer
 * \tparam InStream input streamer, i.e. with input operator
 * \param[in,out] in_str input streamer, e.g. \a std::cin
 * \param[out] opt control parameters
 * \return reference to \a in_str to enable chain reaction
 * \ingroup cpp
 * \note Read data in sequential order with default separators
 */
template <class InStream>
inline InStream &operator>>(InStream &in_str, Options &opt) {
  in_str >> opt.tau_L >> opt.tau_U >> opt.tau_d >> opt.tau_kappa >>
      opt.alpha_L >> opt.alpha_U >> opt.rho >> opt.c_d >> opt.c_h >> opt.N >>
      opt.verbose >> opt.rf_par >> opt.reorder >> opt.saddle;
  return in_str;
}

/*!
 * \brief get the default configuration
 * \ingroup cpp
 */
inline Options get_default_options() { return ::psmilu_get_default_options(); }

/*!
 * \brief represent an option control with C++ string
 * \param[in] opt input option controls
 * \return string representation of \a opt
 * \ingroup cpp
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
         pack_double("tau_d", opt.tau_d) +
         pack_double("tau_kappa", opt.tau_kappa) +
         pack_int("alpha_L", opt.alpha_L) + pack_int("alpha_U", opt.alpha_U) +
         pack_double("rho", opt.rho) + pack_double("c_d", opt.c_d) +
         pack_double("c_h", opt.c_h) + pack_int("N", opt.N) +
         pack_name("verbose", get_verbose) + pack_int("rf_par", opt.rf_par) +
         pack_name("reorder", get_reorder_name) +
         pack_int("saddle", opt.saddle);
}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
/*
 * build a byte map, i.e. the value is the leading byte position of the attrs
 * in Options
 */
const static std::size_t option_attr_pos[14] = {
    0,
    sizeof(double),
    option_attr_pos[1] + sizeof(double),
    option_attr_pos[2] + sizeof(double),
    option_attr_pos[3] + sizeof(double),
    option_attr_pos[4] + sizeof(int),
    option_attr_pos[5] + sizeof(int),
    option_attr_pos[6] + sizeof(double),
    option_attr_pos[7] + sizeof(double),
    option_attr_pos[8] + sizeof(double),
    option_attr_pos[9] + sizeof(int),
    option_attr_pos[10] + sizeof(int),
    option_attr_pos[11] + sizeof(int),
    option_attr_pos[12] + sizeof(int)};

/* data type tags, true for double, false for int */
const static bool option_dtypes[14] = {true,  true,  true,  true, false,
                                       false, true,  true,  true, false,
                                       false, false, false, false};

/* using unordered map to store the string to index map */
const static std::unordered_map<std::string, int> option_tag2pos = {
    {"tau_L", 0},    {"tau_U", 1},   {"tau_d", 2},    {"tau_kappa", 3},
    {"alpha_L", 4},  {"alpha_U", 5}, {"rho", 6},      {"c_d", 7},
    {"c_h", 8},      {"N", 9},       {"verbose", 10}, {"rf_par", 11},
    {"reorder", 12}, {"saddle", 13}};

} /* namespace internal */
#  endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// \brief set \ref Options attribute value from key value pairs
/// \tparam T value type, either \a double or \a int
/// \param[in] attr attribute/member name
/// \param[in] v value
/// \param[out] opt output options
/// \ingroup cpp
///
/// This function can be handy while initialing option parameters from string
/// values. Notice that the keys (string values) are the same as the attribute
/// variable names.
template <typename T>
inline bool set_option_attr(const std::string &attr, const T v, Options &opt) {
  const static bool failed  = true;
  char *            opt_raw = reinterpret_cast<char *>(&opt);
  try {
    const int         pos     = internal::option_tag2pos.at(attr);
    const std::size_t pos_raw = internal::option_attr_pos[pos];
    const bool        dtype   = internal::option_dtypes[pos];
    opt_raw += pos_raw;
    if (dtype)
      *reinterpret_cast<double *>(opt_raw) = v;
    else
      *reinterpret_cast<int *>(opt_raw) = v;
    return !failed;
  } catch (const std::out_of_range &) {
    return failed;
  }
}

} /* namespace psmilu */

/*!
 * \def psmilu_verbose2(__LVL, __opt_tag)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 */
#  define psmilu_verbose2(__LVL, __opt_tag) \
    (__opt_tag & ::psmilu::VERBOSE_##__LVL)

/*!
 * \def psmilu_verbose(__LVL, __opt)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 *
 * \code{.cpp}
 * if (psmilu_verbose(INFO, opt)) ...;
 * \endcode
 */
#  define psmilu_verbose(__LVL, __opt) psmilu_verbose2(__LVL, __opt.verbose)

std::string psmilu::get_verbose(const psmilu::Options &opt) {
  std::string name("");
  if (psmilu_verbose(NONE, opt))
    name = "none";
  else {
    if (psmilu_verbose(INFO, opt)) name = "info";
    if (psmilu_verbose(PRE, opt)) {
      if (name != "")
        name += ",pre";
      else
        name = "pre";
    }
    if (psmilu_verbose(FAC, opt)) {
      if (name != "")
        name += ",fac";
      else
        name = "fac";
    }
    if (psmilu_verbose(MEM, opt)) {
      if (name != "")
        name += ",mem";
      else
        name = "mem";
    }
  }
  return name;
}

#endif /* __cplusplus */

#endif /* _PSMILU_OPTIONS_H */
