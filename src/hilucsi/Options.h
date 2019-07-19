/*
//@HEADER
//----------------------------------------------------------------------------
//                Copyright (C) 2019 The HILUCSI AUTHORS
//----------------------------------------------------------------------------
//@HEADER
*/

/*!
 * \file hilucsi/Options.h
 * \brief HILUCSI algorithm parameter controls
 * \authors Qiao,
 */

#ifndef _HILUCSI_OPTIONS_H
#define _HILUCSI_OPTIONS_H

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
  HILUCSI_VERBOSE_NONE = 0,                         /*!< mute */
  HILUCSI_VERBOSE_INFO = 1,                         /*!< general information */
  HILUCSI_VERBOSE_PRE  = HILUCSI_VERBOSE_INFO << 1, /*!< preprocessing */
  HILUCSI_VERBOSE_FAC  = HILUCSI_VERBOSE_PRE << 1,  /*!< factorization update */
  HILUCSI_VERBOSE_PRE_TIME = HILUCSI_VERBOSE_FAC << 1, /*! pre time */
  HILUCSI_VERBOSE_MEM      = HILUCSI_VERBOSE_PRE_TIME << 1,
  /*!< memory debug */
};

/*!
 * \brief reordering method
 */
enum {
  HILUCSI_REORDER_OFF  = 0, /*!< turn reordering off */
  HILUCSI_REORDER_AUTO = 1, /*!< use automaticaly reordering (default) */
  HILUCSI_REORDER_AMD  = 2, /*!< use AMD ordering */
  HILUCSI_REORDER_RCM  = 3, /*!< use RCM ordering (require BGL) */
  HILUCSI_REORDER_NULL = 6, /*!< ordering Null flag (internal checking) */
};

/*!
 * \struct hilucsi_Options
 * \brief POD parameter controls
 * \note Values in parentheses are default settings
 */
struct hilucsi_Options {
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
  int    check;     /*!< check user input (default is true (!=0)) */
  int    pre_scale; /*!< prescale (default 0 (off)) */
  int    symm_pre_lvls;
  /*!< levels to be applied with symm preprocessing (default is 1) */
  int threads; /*!< user specified threads (default 0) */
};

/*!
 * \typedef hilucsi_Options
 * \brief type wrapper
 */
typedef struct hilucsi_Options hilucsi_Options;

/*!
 * \brief get the default controls
 * \note See the values of attributes in parentheses
 */
static hilucsi_Options hilucsi_get_default_options(void) {
  return (hilucsi_Options){.tau_L         = 0.0001,
                           .tau_U         = 0.0001,
                           .tau_d         = 3.0,
                           .tau_kappa     = 3.0,
                           .alpha_L       = 10,
                           .alpha_U       = 10,
                           .rho           = 0.5,
                           .c_d           = 10.0,
                           .c_h           = 2.0,
                           .N             = -1,
                           .verbose       = HILUCSI_VERBOSE_INFO,
                           .rf_par        = 1,
                           .reorder       = HILUCSI_REORDER_AUTO,
                           .saddle        = 1,
                           .check         = 1,
                           .pre_scale     = 0,
                           .symm_pre_lvls = 1,
                           .threads       = 0};
}

/*!
 * \brief enable verbose flag
 * \param[in] flag verbose flag
 * \param[in,out] opts options
 */
static void hilucsi_enable_verbose(const int flag, hilucsi_Options *opts) {
  if (opts) {
    if (flag > 0)
      opts->verbose |= flag;
    else
      opts->verbose = HILUCSI_VERBOSE_NONE;
  }
}

/*!
 * \brief get the string tag for reordering methods
 * \param[in] opt options
 * \return C-string of the method name
 */
static const char *hilucsi_get_reorder_name(const hilucsi_Options *opt) {
  if (opt) {
    switch (opt->reorder) {
      case HILUCSI_REORDER_OFF:
        return "Off";
      case HILUCSI_REORDER_AUTO:
        return "Auto";
      case HILUCSI_REORDER_AMD:
        return "AMD";
      case HILUCSI_REORDER_RCM:
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

namespace hilucsi {

/*!
 * \addtogroup itr
 * @{
 */

/*!
 * \brief enum wrapper
 * \note The prefix of \a HILUCSI will be dropped
 */
enum : int {
  VERBOSE_NONE     = ::HILUCSI_VERBOSE_NONE,     /*!< mute */
  VERBOSE_INFO     = ::HILUCSI_VERBOSE_INFO,     /*!< general information */
  VERBOSE_PRE      = ::HILUCSI_VERBOSE_PRE,      /*!< preprocessing */
  VERBOSE_FAC      = ::HILUCSI_VERBOSE_FAC,      /*!< factorization update */
  VERBOSE_PRE_TIME = ::HILUCSI_VERBOSE_PRE_TIME, /*!< pre time */
  VERBOSE_MEM      = ::HILUCSI_VERBOSE_MEM,      /*!< memory debug */
};

/*!
 * \brief enum wrapper for reordering methods
 * \note The prefix of \a HILUCSI will be dropped
 */
enum : int {
  REORDER_OFF  = ::HILUCSI_REORDER_OFF, /*!< turn reordering off */
  REORDER_AUTO = ::HILUCSI_REORDER_AUTO,
  /*!< use automaticaly reordering (default) */
  REORDER_AMD  = ::HILUCSI_REORDER_AMD,  /*!< use AMD ordering */
  REORDER_RCM  = ::HILUCSI_REORDER_RCM,  /*!< use RCM ordering (require BGL) */
  REORDER_NULL = ::HILUCSI_REORDER_NULL, /*!< ordering Null flag */
};

/*!
 * \typedef Options
 * \brief type wrapper
 */
typedef hilucsi_Options Options;

/*!
 * \brief get the reordering method name
 */
inline std::string get_reorder_name(const Options &opt) {
  return ::hilucsi_get_reorder_name(&opt);
}

/*!
 * \brief enable verbose flags
 * \param[in] flag verbose options
 * \param[in,out] opts options
 */
inline void enable_verbose(const int flag, Options &opts) {
  ::hilucsi_enable_verbose(flag, &opts);
}

/*!
 * \brief get the verbose name
 */
inline std::string get_verbose(const Options &opt);

/*!
 * \brief read control parameters from a standard input streamer
 * \tparam InStream input streamer, i.e. with input operator
 * \param[in,out] in_str input streamer, e.g. \a std::cin
 * \param[out] opt control parameters
 * \return reference to \a in_str to enable chain reaction
 * \note Read data in sequential order with default separators
 */
template <class InStream>
inline InStream &operator>>(InStream &in_str, Options &opt) {
  in_str >> opt.tau_L >> opt.tau_U >> opt.tau_d >> opt.tau_kappa >>
      opt.alpha_L >> opt.alpha_U >> opt.rho >> opt.c_d >> opt.c_h >> opt.N >>
      opt.verbose >> opt.rf_par >> opt.reorder >> opt.saddle >> opt.check >>
      opt.pre_scale >> opt.symm_pre_lvls >> opt.threads;
  return in_str;
}

/*!
 * \brief get the default configuration
 */
inline Options get_default_options() { return ::hilucsi_get_default_options(); }

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
         pack_double("tau_d", opt.tau_d) +
         pack_double("tau_kappa", opt.tau_kappa) +
         pack_int("alpha_L", opt.alpha_L) + pack_int("alpha_U", opt.alpha_U) +
         pack_double("rho", opt.rho) + pack_double("c_d", opt.c_d) +
         pack_double("c_h", opt.c_h) + pack_int("N", opt.N) +
         pack_name("verbose", get_verbose) + pack_int("rf_par", opt.rf_par) +
         pack_name("reorder", get_reorder_name) +
         pack_int("saddle", opt.saddle) +
         pack_name(
             "check",
             [](const Options &opt_) { return opt_.check ? "yes" : "no"; }) +
         pack_int("pre_scale", opt.pre_scale) +
         pack_int("symm_pre_lvls", opt.symm_pre_lvls) +
         pack_int("threads", opt.threads);
}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace internal {
#    define _HILUCSI_TOTAL_OPTIONS 18
/*
 * build a byte map, i.e. the value is the leading byte position of the attrs
 * in Options
 */
const static std::size_t option_attr_pos[_HILUCSI_TOTAL_OPTIONS] = {
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
    option_attr_pos[12] + sizeof(int),
    option_attr_pos[13] + sizeof(int),
    option_attr_pos[14] + sizeof(int),
    option_attr_pos[15] + sizeof(int),
    option_attr_pos[16] + sizeof(int)};

/* data type tags, true for double, false for int */
const static bool option_dtypes[_HILUCSI_TOTAL_OPTIONS] = {
    true,  true,  true,  true,  false, false, true,  true,  true,
    false, false, false, false, false, false, false, false, false};

/* using unordered map to store the string to index map */
const static std::unordered_map<std::string, int> option_tag2pos = {
    {"tau_L", 0},
    {"tau_U", 1},
    {"tau_d", 2},
    {"tau_kappa", 3},
    {"alpha_L", 4},
    {"alpha_U", 5},
    {"rho", 6},
    {"c_d", 7},
    {"c_h", 8},
    {"N", 9},
    {"verbose", 10},
    {"rf_par", 11},
    {"reorder", 12},
    {"saddle", 13},
    {"check", 14},
    {"pre_scale", 15},
    {"symm_pre_lvls", 16},
    {"threads", 17}};

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

/*!
 * @}
 */

} /* namespace hilucsi */

/*!
 * \def hilucsi_verbose2(__LVL, __opt_tag)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 */
#  define hilucsi_verbose2(__LVL, __opt_tag) \
    (__opt_tag & ::hilucsi::VERBOSE_##__LVL)

/*!
 * \def hilucsi_verbose(__LVL, __opt)
 * \brief return \a true if certain verbose level is defined
 * \note __LVL must be upper case and align with the enumerators
 * \note This macro is for algorithm implementation thus available only in C++
 * \ingroup util
 *
 * \code{.cpp}
 * if (hilucsi_verbose(INFO, opt)) ...;
 * \endcode
 */
#  define hilucsi_verbose(__LVL, __opt) hilucsi_verbose2(__LVL, __opt.verbose)

std::string hilucsi::get_verbose(const hilucsi::Options &opt) {
  std::string name("");
  if (opt.verbose == VERBOSE_NONE)
    name = "none";
  else {
    if (hilucsi_verbose(INFO, opt)) name = "info";
    if (hilucsi_verbose(PRE, opt)) {
      if (name != "")
        name += ",pre";
      else
        name = "pre";
    }
    if (hilucsi_verbose(FAC, opt)) {
      if (name != "")
        name += ",fac";
      else
        name = "fac";
    }
    if (hilucsi_verbose(PRE_TIME, opt)) {
      if (name != "")
        name += ",pre_time";
      else
        name = "pre_time";
    }
    if (hilucsi_verbose(MEM, opt)) {
      if (name != "")
        name += ",mem";
      else
        name = "mem";
    }
  }
  return name;
}

#endif /* __cplusplus */

#endif /* _HILUCSI_OPTIONS_H */
