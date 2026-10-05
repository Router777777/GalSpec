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

    _ARRAYS = ("samples", "samples_id", "logwt", "logvol", "logl", "logz", "logzerr",
               "information")

    def __init__(self, stub, nlive=None, logz_final=None, niter=None):
        for key in self._ARRAYS:
            value = getattr(stub, key, None)
            if value is not None:
                setattr(self, key, np.asarray(value))
        missing = [key for key in ("samples", "logwt", "logz") if not hasattr(self, key)]
        if missing:
            raise RuntimeError(
                "restored results are missing {}; cannot plot".format(", ".join(missing)))
        self.nlive = int(nlive) if nlive is not None else len(self.samples)
        self._niter = int(niter) if niter is not None else None

        # The weights need the *final* evidence as the normalisation. With thinning
        # the last stored logz is an earlier iteration, so prefer the exact value
        # from the file's LOG_EVIDENCE keyword.
        if logz_final is not None and np.isfinite(logz_final):
            self._logz_final = float(logz_final)
        else:
            self._logz_final = float(self.logz[-1])

        # Files written before the per-iteration arrays were thinned together store
        # logz/logzerr at full length; put them back on the sample grid so runplot
        # can pair them with -logvol. Both endpoints are shared by the interpolation,
        # so logz[-1] survives it.
        nsamples = len(self.samples)
        for key in ("logz", "logzerr"):
            array = getattr(self, key, None)
            if array is not None and len(array) != nsamples:
                grid = np.linspace(0.0, 1.0, nsamples)
                setattr(self, key, np.interp(grid, np.linspace(0.0, 1.0, len(array)), array))

    @property
    def niter(self):
        # `runplot` builds its live-points panel as `ones(niter) * nlive` and plots it
        # against -logvol, which is `len(samples)` long. So it only accepts a niter
        # satisfying `niter == nsamps` (flat panel) or `nsamps - niter == nlive`
        # (flat panel plus the descending final-live-points tail); any other value
        # raises a shape error. A full-resolution file meets the second identity
        # exactly, so its stored count is used as-is. A thinned file meets neither,
        # and the count the stored subsample represents is reported instead -- which
        # keeps the panel level right and lets `mark_final_live` still mark the tail.
        if self._niter is not None and self._niter + self.nlive == len(self.samples):
            return self._niter
        return max(1, len(self.samples) - self.nlive)

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
        weights = np.exp(self.logwt - self._logz_final)
        return weights / np.sum(weights)


class _EqualWeightResults:
    """Present ``samples_equal_weight`` in the shape ``dynesty.plotting`` expects.

    ``resample_equal`` draws every row of ``samples_equal_weight`` with the same
    weight, so those samples *are* the posterior and a plot that needs nothing but
    the posterior can read them directly instead of going through ``self.results``
    and its importance weights. Only ``samples``/``importance_weights()`` are
    supplied: the run history (``logvol``, ``logz``, ``niter``, ...) is deliberately
    absent, because resampling shuffles the samples and so no longer pairs a sample
    with its own point along the chain.
    """

    def __init__(self, samples):
        self.samples = np.asarray(samples)

    def importance_weights(self):
        return np.full(len(self.samples), 1.0 / len(self.samples))

    def keys(self):
        return ["samples"]

    def __getitem__(self, key):
        if key == "samples":
            return self.samples
        raise KeyError(key)

    def __contains__(self, key):
        return key == "samples"


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

    # Plot geometry in inches, shared by the plotting methods.
    # A corner plot grows as n^2 panels and a trace plot as n rows, so a fixed
    # figure squeezes them until the ticks and titles collide.
    #: Panel edge that keeps ticks and labels legible.
    _MIN_PANEL_SIZE = 1.1
    #: Panel edge used when the corner figure size is derived automatically.
    _CORNER_PANEL_SIZE = 2.2

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

    def _resolve_params(self, params):
        """Turn a ``params`` selection into indices into :attr:`param_names`.

        ``None`` selects every parameter, in the fitted order. Otherwise ``params``
        is a sequence naming the parameters to keep, in the order they are to be
        plotted; each entry is either an index or a parameter name, and the two
        styles can be mixed -- ``params=[0, 1, 'sigma_c_7']``. Names are matched
        against :attr:`param_names`, then against :attr:`full_param_names`, then
        against the readable labels from :meth:`parameter_labels`, so
        ``'sigma_c_7'``, ``'broad Halpha.sigma_c'`` and
        ``'broad Halpha sigma_c'`` all work.
        """
        if params is None:
            return list(range(self.ndim))

        indices = []
        for entry in params:
            if isinstance(entry, str):
                candidates = (self.param_names, self.full_param_names,
                              self.parameter_labels(style='compound'))
                for names in candidates:
                    if entry in names:
                        indices.append(names.index(entry))
                        break
                else:
                    raise ValueError(
                        "unknown parameter {!r}; the model has {}".format(
                            entry, ", ".join(self.full_param_names)))
                continue
            index = int(entry)
            if index < 0:                      # allow the usual negative indexing
                index += self.ndim
            if not 0 <= index < self.ndim:
                raise IndexError(
                    "parameter index {} is out of range for {} parameters".format(
                        entry, self.ndim))
            indices.append(index)
        return indices

    #: How the plotting methods label parameters.
    #:   ``'compound'`` -> ``'narrow Halpha amplitude'``
    #:   ``'dot'``      -> ``'narrow Halpha.amplitude'``  (as in full_param_names)
    #:   ``'index'``    -> ``'amplitude_1'``             (the raw sampler name)
    _LABEL_STYLES = ('compound', 'dot', 'index')

    def _compound_name(self, param_name):
        """Split a compound parameter name into ``(component, parameter)``.

        ``'amp_c_2'`` -> ``('broad Halpha', 'amp_c')``. The trailing index selects
        the submodel directly, so this keeps working even when two components share
        a name or the names were auto-generated.
        """
        base = self._base_name(param_name)
        suffix = param_name[len(base) + 1:] if param_name.startswith(base + '_') else ''
        if suffix.isdigit():
            submodel_names = list(getattr(self.model, 'submodel_names', ()) or ())
            index = int(suffix)
            if 0 <= index < len(submodel_names):
                return submodel_names[index], base
        return None, base

    def parameter_labels(self, index=None, style='compound'):
        """Readable labels for the parameters at ``index``.

        The name given to the component when it was built is combined with the
        parameter's own name, so a parameter the sampler reports as ``amplitude_1``
        on a component named ``'narrow Halpha'`` is labelled
        ``'narrow Halpha amplitude'``.
        """
        if style not in self._LABEL_STYLES:
            raise ValueError("label_style must be one of {}; got {!r}".format(
                ", ".join(self._LABEL_STYLES), style))
        if index is None:
            index = range(self.ndim)
        labels = []
        for i in index:
            name = self.param_names[i]
            if style == 'index':
                labels.append(name)
                continue
            component, base = self._compound_name(name)
            if component is None:
                labels.append(name)
            elif style == 'dot':
                labels.append("{}.{}".format(component, base))
            else:
                labels.append("{} {}".format(component, base))
        return labels

    def _plot_labels(self, index, labels, label_style):
        """Resolve the labels for a plot of the parameters at ``index``."""
        if labels is None:
            return self.parameter_labels(index, style=label_style)
        labels = list(labels)
        if len(labels) != len(index):
            raise ValueError("got {} labels for {} parameters".format(
                len(labels), len(index)))
        return labels

    def _corner_figsize(self, nplot, figsize=None):
        """Return the figure size for a corner plot of ``nplot`` parameters.

        With ``figsize=None`` the size is derived from the panel count, so every
        panel keeps the same room for its ticks and titles however many parameters
        are selected. An explicit ``figsize`` is passed through, after warning when
        its panels are too small to stay legible.
        """
        if figsize is not None:
            if figsize[0] / nplot < self._MIN_PANEL_SIZE:
                warnings.warn(
                    "figsize gives {:.2f} in per parameter panel, but {} parameters "
                    "are plotted; below ~{:.1f} in the ticks and titles collide. "
                    "Pass figsize=None to size the figure automatically.".format(
                        figsize[0] / nplot, nplot, self._MIN_PANEL_SIZE))
            return figsize

        side = self._CORNER_PANEL_SIZE * nplot
        return (side, side)

    def plot_corner(self, params=None, figsize=None, labels=None,
                    label_style='compound', **kwargs):
        """Corner plot drawn by the ``corner`` package.

        ``params`` selects which parameters to plot, as in :meth:`_resolve_params`.
        ``figsize=None`` sizes the figure from the number of plotted parameters, as
        in :meth:`_corner_figsize`. ``labels`` overrides the axis labels outright;
        by default they come from :meth:`parameter_labels` and name the component
        each parameter belongs to, in the style set by ``label_style``.
        """
        if (corner is None) or (plt is None):
            raise ImportError("corner and matplotlib is required for plotting")
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        if self.theta_best is None:
            self.get_best_fit()
        index = self._resolve_params(params)
        labels = self._plot_labels(index, labels, label_style)
        samples = self.samples_equal_weight[:, index]
        # corner has no `figsize` argument -- it would be swallowed by its **kwargs
        # and the size ignored -- so the figure is created here and handed to it.
        fig = plt.figure(figsize=self._corner_figsize(len(index), figsize))
        if len(index) != 1:
            return corner.corner(samples, labels=labels, fig=fig,
                                 truths=self.theta_best[index], **kwargs)

        # A one-parameter corner plot cannot take `truths`: corner 2.2.3's
        # `_get_fig_axes` returns a bare Axes for a single panel, which
        # `overplot_lines` then indexes as `axes[k1, k1]` and raises "'Axes' object
        # is not subscriptable". Suppress it and draw the same line here; the square
        # marker corner adds along with it is skipped for one parameter anyway.
        truth = self.theta_best[index][0]
        truth_color = kwargs.pop('truth_color', '#4682b4')     # corner's own default
        fig = corner.corner(samples, labels=labels, fig=fig,
                            truths=None, **kwargs)
        if truth is not None:
            fig.axes[0].axvline(truth, color=truth_color)
        return fig

    def _results_for_plotting(self):
        """Return a results object usable by ``dynesty.plotting``.

        A live fit already holds a real :class:`dynesty.results.Results`. A fit
        restored by :func:`load_dynesty_fit_from_fits` holds only plain arrays, so
        wrap them in :class:`_ReloadedResults` to expose the interface ``dyplot``
        needs (item access plus ``importance_weights``).

        For the run-history diagnostics (:meth:`plot_summary`, :meth:`plot_trace`),
        which need ``logvol``/``logz``/``niter`` and not just the posterior. A plot
        that needs only the posterior should use :class:`_EqualWeightResults`
        instead, which does not depend on the raw sampler output surviving.
        """
        if self.results is None:
            raise RuntimeError("fit() must be run first")
        if hasattr(self.results, "importance_weights"):
            return self.results
        return _ReloadedResults(self.results, nlive=self.nlive,
                                logz_final=self.log_evidence,
                                niter=getattr(self.results, "niter", None))

    def plot_corner_dyplot(self, params=None, figsize=None, labels=None,
                           label_style='compound'):
        """Corner plot drawn by ``dynesty.plotting.cornerplot``.

        Drawn from ``samples_equal_weight``, the same resampled posterior as
        :meth:`plot_corner`, so both corner plots show one posterior; ``self.results``
        is not needed and the plot therefore also works on a restored fit.

        ``params`` selects which parameters to plot, as in :meth:`_resolve_params`.
        ``figsize=None`` sizes the figure from the number of plotted parameters, as
        in :meth:`_corner_figsize`. ``labels`` overrides the axis labels outright;
        by default they come from :meth:`parameter_labels` and name the component
        each parameter belongs to, in the style set by ``label_style``.
        """
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        if self.samples_equal_weight is None:
            raise RuntimeError("fit() must be run first")
        index = self._resolve_params(params)
        labels = self._plot_labels(index, labels, label_style)
        results = _EqualWeightResults(self.samples_equal_weight)
        if self.theta_best is None:
            self.get_best_fit()

        # The panel grid must hold one row and column per plotted parameter, while
        # `dims` slices the samples inside cornerplot -- so `labels` and `truths`
        # have to be sliced here to stay aligned with it.
        figsize = self._corner_figsize(len(index), figsize)
        fig, axes = plt.subplots(len(index), len(index), figsize=figsize)

        fig_corner, axes_corner = dyplot.cornerplot(
            results,
            fig=(fig, axes),
            dims=index,
            labels=labels,
            truths=self.theta_best[index],
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

    def plot_summary(self, figsize=(16, 16), logplot=True):
        """Run diagnostics drawn by ``dynesty.plotting.runplot``.

        ``logplot=True`` (the default) plots log Z in the evidence panel. Without it
        that panel is degenerate for any real fit: the evidence is a few hundred in
        magnitude, so ``exp(log Z)`` underflows to zero and the panel is a flat line
        at 0 (which also triggers runplot's "identical ylims" warning).
        """
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        results = self._results_for_plotting()

        fig_summary, axes_summary = dyplot.runplot(
            results,
            color='C3',
            lnz_error=True,
            mark_final_live=True,
            logplot=logplot,
        )
        # The figure is resized here rather than passed to runplot as `fig=`. When
        # given a figure, runplot reads its y-limits off the blank axes it receives
        # and then only pins the top, which collapses the evidence panel to an
        # inverted 0..logZ axis and clips the whole log Z curve off the plot.
        fig_summary.set_size_inches(*figsize)
        return fig_summary, axes_summary

    def plot_trace(self, params=None, figsize=None, labels=None,
                   label_style='compound'):
        """Trace plot drawn by ``dynesty.plotting.traceplot``.

        ``params`` selects which parameters to plot, as in :meth:`_resolve_params`.
        ``figsize=None`` sizes the figure from the number of plotted parameters at
        ~2.4 in per row, the minimum height that keeps each row's title and labels
        clear of the row above. ``labels`` overrides the axis labels outright; by
        default they come from :meth:`parameter_labels` and name the component each
        parameter belongs to, in the style set by ``label_style``.
        """
        if (dyplot is None) or (plt is None):
            raise ImportError("dynesty and matplotlib is required for plotting")
        index = self._resolve_params(params)
        labels = self._plot_labels(index, labels, label_style)
        results = self._results_for_plotting()
        nrows = len(index)
        if figsize is None:
            figsize = (10, 2.4 * nrows)
        elif figsize[1] / nrows < self._MIN_PANEL_SIZE:
            # The figure always gets one row per plotted parameter, so a height sized
            # for fewer rows squeezes them until the labels collide.
            warnings.warn(
                "figsize gives {:.2f} in per parameter row, but {} parameters are "
                "plotted; below ~{:.1f} in per row the labels collide. Pass "
                "figsize=None to size the figure automatically.".format(
                    figsize[1] / nrows, nrows, self._MIN_PANEL_SIZE))

        fig_trace, axes_trace = plt.subplots(nrows, 2, figsize=figsize,
                                     gridspec_kw={'hspace': 0.45})
        # `bottom` is a fraction of the figure height, so a fixed 0.05 starves short
        # figures of the absolute margin the bottom row's x-label needs.
        bottom = max(0.05, 0.55 / figsize[1])
        fig_trace.subplots_adjust(top=0.985, bottom=bottom, left=0.07, right=0.985)

        fig_trace, axes_trace = dyplot.traceplot(
            results,
            # `labels` must list the plotted parameters only, in the order given by
            # `dims`, since traceplot slices the samples but not the labels.
            fig=(fig_trace, axes_trace),          # must hold exactly len(labels) x 2 panels
            dims=index,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            # show_titles=True would print median_{-a}^{+b} above each right panel, but it needs
            # ~3 in of row height to clear the row above. The quantile table has the numbers.
            show_titles=False,
            thin=5,
            max_n_ticks=3,
            label_kwargs={'fontsize': 8},
        )
        return fig_trace, axes_trace

    
