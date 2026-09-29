"""Nested-sampling fits for Astropy compound models."""

from __future__ import annotations

import time
import warnings
from copy import deepcopy

import numpy as np
from astropy.modeling import CompoundModel

try:
    import dynesty
    from dynesty import utils as dyfunc
except ImportError:  # pragma: no cover
    dynesty = None
    dyfunc = None

try:
    from dynesty.pool import Pool as DynestyPool
except ImportError:  # pragma: no cover - dynesty before its native pool
    DynestyPool = None

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

try: 
    from dynesty import plotting as dyplot
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    dyplot = None
    plt = None

__all__ = ["Dynesty_Fit"]

_WORKER_FIT = None


def _initialize_likelihood_worker(fit):
    """Install one fitter copy per worker, avoiding per-call model serialization."""
    global _WORKER_FIT
    _WORKER_FIT = fit


def _worker_log_likelihood(theta):
    return _WORKER_FIT.log_likelihood(theta)


class _LikelihoodPool:
    """Legacy dynesty 2.x adapter for a multiprocess pool."""

    def __init__(self, pool):
        self._pool = pool

    def map(self, ignored_function, values):
        if getattr(ignored_function, "__name__", None) is not None:
            return self._pool.map(ignored_function, values)
        return self._pool.map(_worker_log_likelihood, values)


class _ReloadedResults:
    """Give a FITS-restored results stub the interface ``dynesty.plotting`` expects.

    ``load_dynesty_fit_from_fits`` stores plain arrays on a bare container, whereas
    ``dyplot.cornerplot``/``runplot``/``traceplot`` index the results object
    (``results['logvol']``) and call ``results.importance_weights()``. This adapter
    supplies exactly those, so the plotting methods also work on a restored fit.

    Note the RAW samples are used rather than ``samples_equal_weight``: ``traceplot``
    pairs every sample with its own ``logvol``, and ``resample_equal`` deliberately
    returns its output in random order, which would destroy that pairing.
    """

    _ARRAYS = ("samples", "logwt", "logvol", "logl", "logz", "logzerr", "information")

    def __init__(self, stub, nlive=None):
        for key in self._ARRAYS:
            value = getattr(stub, key, None)
            if value is not None:
                setattr(self, key, np.asarray(value))
        missing = [key for key in ("samples", "logwt", "logz") if not hasattr(self, key)]
        if missing:
            raise RuntimeError(
                "restored results are missing {}; cannot plot".format(", ".join(missing)))
        self.nlive = int(nlive) if nlive is not None else len(self.samples)

        # `save_dynesty_fit_to_fits` thins the per-iteration arrays (samples, logwt,
        # logl, logvol) but writes logz/logzerr in full, and `runplot` plots logz
        # against -logvol. Put them back on the sample grid; both endpoints are
        # shared, so logz[-1] -- which importance_weights() relies on -- is exact.
        nsamples = len(self.samples)
        for key in ("logz", "logzerr"):
            array = getattr(self, key, None)
            if array is not None and len(array) != nsamples:
                grid = np.linspace(0.0, 1.0, nsamples)
                setattr(self, key, np.interp(grid, np.linspace(0.0, 1.0, len(array)), array))

    @property
    def niter(self):
        # `runplot` sizes its live-points panel from niter. After thinning, no array
        # records the iteration count, and the final live points cannot be identified
        # inside a thinned array, so report the stored sample count (runplot then
        # disables `mark_final_live` with a warning, which is the honest outcome).
        return len(self.samples)

    def keys(self):
        return [key for key in self._ARRAYS if hasattr(self, key)]

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key):
        return hasattr(self, key)

    def importance_weights(self):
        # Normalised: thinning leaves the raw exp(logwt - logz[-1]) summing to ~1/thin,
        # which is not a probability vector over the stored samples.
        weights = np.exp(self.logwt - self.logz[-1])
        return weights / np.sum(weights)


class Dynesty_Fit:
    """Fit an Astropy compound model with :mod:`dynesty`.

    Model bounds are used whenever both ends are finite. Otherwise a conservative
    parameter-type default is used. Default amplitude priors are log-uniform;
    explicitly supplied bounds are always linear-uniform.

    ``n_processes > 1`` parallelizes likelihood calls. Alternatively, ``pool``
    accepts a user-managed object with a ``map`` method; user pools are not closed.
    Invalid samples and samples with non-positive uncertainty are ignored.
    """

    DEFAULT_BOUNDS = {
        "amplitude": (1e-6, 1e6),
        "dv": (-10000.0, 10000.0),
        "sigma": (10.0, 20000.0),
        "stddev": (10.0, 20000.0),
        "h3": (-0.5, 0.5),
        "h4": (-0.5, 0.5),
        "logtau": (-3.0, 3.0),
        "cf": (0.0, 1.0),
        "wavec": (0.99, 1.01),
        "continuum": (-10.0, 10.0),
        "alpha": (-5.0, 5.0),
        "redshift": (-0.1, 0.1),
        "velscale": (10.0, 1000.0),
    }

    def __init__(self, model, wave_use, flux_use, ferr, bounds_dict=None,
                 default_bounds=None, sample_method="rwalk", nlive=500,
                 bound="multi", rstate=None, n_processes=1, pool=None,
                 queue_size=None):
        if dynesty is None:
            raise ImportError("dynesty is required; install GalSpec's dependencies")
        if not isinstance(model, CompoundModel):
            raise TypeError("model must be an Astropy CompoundModel")
        if pool is not None and n_processes != 1:
            raise ValueError("pass either pool or n_processes, not both")
        if int(n_processes) < 1:
            raise ValueError("n_processes must be at least 1")

        wave = np.asarray(wave_use, dtype=float)
        flux = np.asarray(flux_use, dtype=float)
        error = np.asarray(ferr, dtype=float)
        if wave.shape != flux.shape or wave.shape != error.shape:
            raise ValueError("wave_use, flux_use, and ferr must have the same shape")
        valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
        if not np.any(valid):
            raise ValueError("no valid data remain after filtering")
        if not np.all(valid):
            warnings.warn(f"Ignoring {valid.size - valid.sum()} invalid data samples")

        self.model = deepcopy(model)
        self.wave_use = wave[valid]
        self.flux_use = flux[valid]
        self.ferr = error[valid]
        self.sample_method = sample_method
        self.nlive = int(nlive)
        self.bound = bound
        self.rstate = rstate
        self.n_processes = int(n_processes)
        self.pool = pool
        self.queue_size = queue_size
        self.bounds_dict = dict(bounds_dict or {})
        self.default_bounds = self.DEFAULT_BOUNDS | dict(default_bounds or {})
        self.custom_priors = {}

        self._name_submodels()
        self.param_names = [name for name in self.model.param_names
                            if not getattr(self.model, name).fixed
                            and not getattr(self.model, name).tied]
        self.full_param_names = self._full_parameter_names()
        self.param_map = dict(zip(self.full_param_names, enumerate(self.param_names)))
        self._log_scale_params = set()
        self.param_bounds = self.get_param_bounds()
        self.theta_initial = np.asarray(
            [getattr(self.model, name).value for name in self.param_names], dtype=float)
        self.ndim = len(self.param_names)
        if self.ndim == 0:
            raise ValueError("model has no free parameters")

        self.results = None
        self.samples_equal_weight = None
        self.log_evidence = None
        self.log_evidence_err = None
        self.theta_best = None
        self.runtime_seconds = None
        self.ncall = None

    def _name_submodels(self):
        for index, submodel in enumerate(self.model):
            if submodel.name is None:
                submodel.name = self.model.submodel_names[index]

    def _full_parameter_names(self):
        names = []
        for model_name in self.model.submodel_names:
            submodel = self.model[model_name]
            for param_name in submodel.param_names:
                param = getattr(submodel, param_name)
                if not param.fixed and not param.tied:
                    names.append(f"{model_name}.{param_name}")
        return names

    @staticmethod
    def _base_name(param_name):
        parts = param_name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else param_name

    def _parameter_type(self, param_name):
        base = self._base_name(param_name).lower()
        if base.startswith("c") and base[1:].isdigit():
            return "continuum"
        if "amplitude" in base or base.startswith("amp"):
            return "amplitude"
        if base.startswith("sigma"):
            return "sigma"
        if base == "stddev":
            return "stddev"
        if base.startswith("dv"):
            return "dv"
        if base.startswith("wavec") or base in {"mean", "x_0"}:
            return "wavec"
        if base.startswith("logtau"):
            return "logtau"
        if base == "cf":
            return "cf"
        if base in {"h3", "h4", "alpha", "velscale"}:
            return base
        if base in {"z", "redshift"}:
            return "redshift"
        return "continuum"

    def _custom_bound(self, param_name, index):
        candidates = [param_name, self._base_name(param_name)]
        if index < len(self.full_param_names):
            candidates.insert(1, self.full_param_names[index])
        for key in candidates:
            if key in self.bounds_dict:
                return self.bounds_dict[key]
        return None

    @staticmethod
    def _validate_bound(name, bound):
        if len(bound) != 2:
            raise ValueError(f"bounds for {name} must contain two values")
        lower, upper = map(float, bound)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"bounds for {name} must be finite and increasing")
        return lower, upper

    def get_param_bounds(self):
        bounds = []
        for index, name in enumerate(self.param_names):
            param = getattr(self.model, name)
            custom = self._custom_bound(name, index)
            if custom is not None:
                bounds.append(self._validate_bound(name, custom))
                continue
            lower, upper = param.bounds
            if lower is not None and upper is not None:
                try:
                    bounds.append(self._validate_bound(name, (lower, upper)))
                    continue
                except ValueError:
                    pass
            kind = self._parameter_type(name)
            default = self._validate_bound(name, self.default_bounds[kind])
            if kind == "wavec":
                initial = float(param.value)
                default = tuple(sorted((initial * default[0], initial * default[1])))
            elif kind == "amplitude":
                default = (np.log10(default[0]), np.log10(default[1]))
                self._log_scale_params.add(name)
            bounds.append(default)
        return bounds

    def prior_transform(self, u):
        u = np.asarray(u, dtype=float)
        theta = np.empty_like(u)
        for index, name in enumerate(self.param_names):
            if name in self.custom_priors:
                theta[index] = self.custom_priors[name](u[index])
            else:
                lower, upper = self.param_bounds[index]
                value = lower + (upper - lower) * u[index]
                theta[index] = 10.0**value if name in self._log_scale_params else value
        return theta

    def set_model_params(self, theta):
        for name, value in zip(self.param_names, theta):
            setattr(self.model, name, value)
        for name in self.model.param_names:
            param = getattr(self.model, name)
            if param.tied:
                param.value = param.tied(self.model)
        return self.model

    def log_likelihood(self, theta):
        try:
            model_flux = np.asarray(self.set_model_params(theta)(self.wave_use), dtype=float)
        except (ArithmeticError, ValueError, RuntimeError):
            return -np.inf
        if model_flux.shape != self.flux_use.shape or not np.all(np.isfinite(model_flux)):
            return -np.inf
        residual = (self.flux_use - model_flux) / self.ferr
        norm = np.log(2.0 * np.pi * self.ferr**2)
        return float(-0.5 * np.sum(residual**2 + norm))

    def fit(self, progress=True, **run_nested_kwargs):
        """Run nested sampling; keywords are forwarded to ``run_nested``."""
        owned_pool = None
        active_pool = self.pool
        loglikelihood = self.log_likelihood
        prior_transform = self.prior_transform
        dynesty_major = int(getattr(dynesty, "__version__", "2").split(".", 1)[0])
        if active_pool is None and self.n_processes > 1:
            if dynesty_major >= 3:
                if DynestyPool is None:  # pragma: no cover
                    raise ImportError("dynesty 3.x pool support is unavailable")
                owned_pool = DynestyPool(
                    self.n_processes, self.log_likelihood, self.prior_transform)
                active_pool = owned_pool.__enter__()
                loglikelihood = active_pool.loglike
                prior_transform = active_pool.prior_transform
            else:  # pragma: no cover - retained for dynesty 2.x
                try:
                    from multiprocess import Pool
                except ImportError as exc:
                    raise ImportError(
                        "multiprocess is required for dynesty 2.x parallel fits") from exc
                raw_pool = Pool(processes=self.n_processes,
                                initializer=_initialize_likelihood_worker,
                                initargs=(self,))
                owned_pool = raw_pool
                active_pool = _LikelihoodPool(raw_pool)

        queue_size = self.queue_size
        if queue_size is None and active_pool is not None:
            queue_size = self.n_processes if owned_pool is not None else 1
        sampler_kwargs = dict(loglikelihood=loglikelihood,
                              prior_transform=prior_transform,
                              ndim=self.ndim, nlive=self.nlive,
                              bound=self.bound, sample=self.sample_method,
                              rstate=self.rstate)
        if active_pool is not None:
            sampler_kwargs.update(
                pool=active_pool,
                queue_size=queue_size,
                use_pool={"prior_transform": False,
                          "loglikelihood": True,
                          "propose_point": True,
                          "update_bound": False},
            )

        started = time.perf_counter()
        try:
            sampler = dynesty.NestedSampler(**sampler_kwargs)
            sampler.run_nested(print_progress=progress, **run_nested_kwargs)
            self.results = sampler.results
        finally:
            if owned_pool is not None:
                if dynesty_major >= 3:
                    active_pool.close()
                    active_pool.join()
                    owned_pool.__exit__(None, None, None)
                else:  # pragma: no cover - dynesty 2.x cleanup
                    owned_pool.close()
                    owned_pool.join()
        self.runtime_seconds = time.perf_counter() - started
        self.ncall = int(np.sum(self.results.ncall))
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        self.samples_equal_weight = dyfunc.resample_equal(
            self.results.samples, weights, rstate=self.rstate)
        self.log_evidence = float(self.results.logz[-1])
        self.log_evidence_err = float(self.results.logzerr[-1])
        return self.samples_equal_weight, self.model, self.param_names

    def get_best_fit(self):
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        self.theta_best = np.asarray(self.results.samples[np.argmax(self.results.logl)])
        self.set_model_params(self.theta_best)
        return self.model, self.param_names, self.theta_best

    def get_quantiles(self, quantiles=(0.16, 0.5, 0.84)):
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        return np.quantile(self.samples_equal_weight, quantiles, axis=0)

    def get_evidence(self):
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        return self.log_evidence, self.log_evidence_err

    def plot_corner(self, **kwargs):
        if corner is None:
            raise ImportError("corner is required for plotting")
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        if self.theta_best is None:
            self.get_best_fit()
        return corner.corner(self.samples_equal_weight, labels=self.param_names,
                             truths=self.theta_best, **kwargs)

    def _results_for_plotting(self):
        """Return a results object usable by ``dynesty.plotting``.

        A live fit already holds a real :class:`dynesty.results.Results`. A fit
        restored by :func:`load_dynesty_fit_from_fits` holds only plain arrays, so
        wrap them in :class:`_ReloadedResults` to expose the interface ``dyplot``
        needs (item access plus ``importance_weights``).
        """
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        if hasattr(self.results, "importance_weights"):
            return self.results
        return _ReloadedResults(self.results, nlive=self.nlive)

    def plot_corner_dyplot(self, figsize=(11, 11)):
        """Corner plot drawn by ``dynesty.plotting.cornerplot``.

        Uses the sampler's own results when they are still in memory; on a restored
        fit it falls back to the arrays stored in the FITS file. For a fallback that
        needs nothing but the resampled posterior, see :meth:`plot_corner`.
        """
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        results = self._results_for_plotting()
        if self.theta_best is None:
            self.get_best_fit()

        fig, axes = plt.subplots(self.ndim, self.ndim, figsize=figsize)

        fig_corner, axes_corner = dyplot.cornerplot(
            results,
            fig=(fig, axes),
            labels=self.param_names,
            truths=self.theta_best,
            color='blue',
            truth_color='black',
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],                #±1σ
            title_quantiles=[0.16, 0.5, 0.84],
            title_fmt='.3g',
            max_n_ticks=3,
            title_kwargs={'y': 1.05, 'fontsize': 6},
            label_kwargs={'fontsize': 7},
            )
        return fig_corner, axes_corner

    def plot_summary(self, figsize=(16, 16)):
        """Run diagnostics drawn by ``dynesty.plotting.runplot``."""
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        results = self._results_for_plotting()

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        fig_summary, axes_summary = dyplot.runplot(
            results,
            fig=(fig, axes),   # default would be a 16x16 in figure
            color='C3',
            lnz_error=True,
            mark_final_live=True,
        )
        return fig_summary, axes_summary

    def plot_trace(self, figsize=None):
        """Trace plot drawn by ``dynesty.plotting.traceplot``.

        ``figsize=None`` sizes the figure from the number of parameters at ~2.4 in
        per row, the minimum height that keeps each row's title and labels clear of
        the row above.
        """
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        results = self._results_for_plotting()
        if figsize is None:
            figsize = (10, 2.4 * self.ndim)
        elif figsize[1] / self.ndim < 1.1:
            # The figure always gets one row per parameter, so a height sized for fewer
            # rows squeezes them until the labels collide.
            warnings.warn(
                "figsize gives {:.2f} in per parameter row, but this model has {} "
                "parameters; below ~1.1 in per row the labels collide. Pass "
                "figsize=None to size the figure automatically.".format(
                    figsize[1] / self.ndim, self.ndim))

        fig_trace, axes_trace = plt.subplots(self.ndim, 2, figsize=figsize,
                                     gridspec_kw={'hspace': 0.45})
        # `bottom` is a fraction of the figure height, so a fixed 0.05 starves short
        # figures of the absolute margin the bottom row's x-label needs.
        bottom = max(0.05, 0.55 / figsize[1])
        fig_trace.subplots_adjust(top=0.985, bottom=bottom, left=0.07, right=0.985)

        fig_trace, axes_trace = dyplot.traceplot(
            results,
            # `dims` omitted on purpose: None means every parameter, and it keeps
            # `labels` in sync with `self.param_names` without index bookkeeping.
            fig=(fig_trace, axes_trace),          # must hold exactly len(labels) x 2 panels
            labels=list(self.param_names),
            quantiles=[0.16, 0.5, 0.84],
            # show_titles=True would print median_{-a}^{+b} above each right panel, but it needs
            # ~3 in of row height to clear the row above. The quantile table has the numbers.
            show_titles=False,
            thin=5,
            max_n_ticks=3,
            label_kwargs={'fontsize': 8},
        )
        return fig_trace, axes_trace

    
