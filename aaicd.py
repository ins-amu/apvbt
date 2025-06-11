#!/usr/bin/env python
# pip install matplotlib vbjax sbi tqdm typed-argparse
# implementation details for "Amortizing Personalization in Virtual Brain Twins."

import argparse
import numpy as np
import pickle
import tqdm
import urllib.request
import tempfile
import zipfile
from collections.abc import Callable
import typed_argparse as tap


def load_pkl(fname):
    with open(fname, 'rb') as fd:
        pkl = pickle.load(fd)
    return pkl

seed = 42
def small(*sh):
    import jax
    return jax.random.normal(
        jax.random.PRNGKey(seed), sh) * 1e-3


def to_torch(x):
    import torch
    return torch.from_numpy(np.array(x)).float()


def triu_to_mat(triu):
    import jax.numpy as jp
    n = triu.shape[1]
    nn = int(jp.ceil(jp.sqrt(n*2)))
    i, j = jp.triu_indices(nn, k=1)
    mat = jp.zeros((triu.shape[0], nn, nn), 'f')
    mat = mat.at[:, i, j].set(triu).at[:, j, i].set(triu)
    return mat


def all_conf_rates(wbs, conns):
    import jax, jax.numpy as jp
    @jax.jit
    def dist(c_d, c_d_h):
        return jp.sum((c_d[:,None] - c_d_h)**2, axis=-1)
    # TODO consider graph & dynamical metrics
    crs = np.zeros((len(conns),)*2)
    for i, (((ew, eb), _), c_e) in enumerate(zip(wbs, conns)):
        u = c_e @ ew + eb  # encode from parc i
        for j, ((_, (dw, db)), c_d) in enumerate(zip(wbs, conns)):
            c_d_h = u @ dw + db
            ok = dist(c_d, c_d_h).argmin(axis=1) == jp.r_[:c_d.shape[0]]
            crs[i, j] = 1 - ok.mean()
    return crs


class MvNorm:
    def __init__(self, us, u_mean, u_cov, key=None):
        import jax
        self.us = us
        self.u_mean = u_mean
        self.u_cov = u_cov
        self.key = key or jax.random.PRNGKey(42)

    def sample(self, n):
        import jax
        self.key, key = jax.random.split(self.key)
        return jax.random.multivariate_normal(
            key, self.u_mean, self.u_cov, shape=(n,))


class XCode:
    "helper for cross-coder"
    wbs = None
    conns = None
    means = None
    parcs = None
    tts = None

    @classmethod
    def from_conns_npz(cls, fname, tts=200):
        import jax.numpy as jp
        KB = np.load(fname, allow_pickle=True)
        parcs = []
        conns = []
        means = []
        for parc in KB.keys():
            if KB[parc].ndim == 3 and parc not in ('031-MIST',):
                parcs.append(parc)
                i, j = jp.triu_indices(KB[parc].shape[1], k=1)
                ctri = KB[parc][:, i, j]
                assert ctri.shape[0] == 274
                ctri = jp.sqrt(ctri)  # Spase scaling
                means.append(ctri.mean(axis=0))
                ctri -= ctri.mean(axis=0)
                conns.append(ctri)
        self = cls()
        self.conns = [jp.array(_.astype('f')) for _ in conns]
        self.means = means
        self.parcs = parcs
        self.tts = tts
        self.wbs = []
        return self

    @classmethod
    def from_kg(cls, zip_fname=None, tts=None, hcp=False, skip_parc=''):
        import jax.numpy as jp
        if zip_fname:
            parsed = cls._parse_counts_from_kg_zip(
                zip_fname, hcp=hcp, skip_parcs=skip_parc)
        else:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                cls._download_kg_zip(temp_file.name, hcp=hcp)
                parsed = cls._parse_counts_from_kg_zip(temp_file.name, hcp=hcp)
        parcs, means, conns = parsed
        self = cls()
        self.conns = [jp.array(_.astype('f')) for _ in conns]
        self.means = [jp.array(_.astype('f')) for _ in means]
        self.parcs = parcs
        self.tts = tts or (conns[0].shape[0] // 2)
        self.wbs = []
        return self

    @staticmethod
    def _download_kg_zip(dl_fname, hcp=False):
        # https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a
        if hcp:
            url = 'https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000059_Atlas_based_HCP_connectomes_v1.1_pub'
        else:
            url = 'https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/public/buckets/d-3f179784-194d-4795-9d8d-301b524ca00a'
        urllib.request.urlretrieve(url, dl_fname)

    @staticmethod
    def _parse_counts_from_kg_zip(zip_fname, hcp=False, skip_parcs=''):
        kb_zip = zipfile.ZipFile(zip_fname)
        parc_zip_fnames = [
            _.filename for _ in kb_zip.filelist if _.filename.endswith('.zip')]
        conns = []
        parcs = []
        means = []
        for l, parc_zip_fname in enumerate(tqdm.tqdm(parc_zip_fnames, ncols=60)):
            parc, _ = parc_zip_fname.split('.zip')
            nreg = int(parc.split('-')[0])
            # if parc in skip_parcs or nreg > 120:
            #     print('skip', parc)
            #     continue
            # if parc in ('031-MIST', '294-Julich-Brain'):  # inconsistent matrices
            #     continue
            parcs.append(parc)
            with kb_zip.open(parc_zip_fname) as parc_zip_fd:
                parc_zip = zipfile.ZipFile(parc_zip_fd)
                ti, tj = np.triu_indices(nreg, k=1)
                ws = np.zeros((200 if hcp else 261, ti.size), 'f')
                for i in range(ws.shape[0]):
                    counts_fname = f'{parc}/SC/{i+1:04d}_1_Counts.csv'
                    if hcp:
                        counts_fname = f'{parc}/1StructuralConnectivity/{i:03d}/Counts.csv'
                    with parc_zip.open(counts_fname) as fd:
                        # np.loadtxt doesn't load from zip fd correctly?
                        txt = fd.read().decode('ascii')
                        delim = ',' if ',' in txt else ' '
                        mat = [[float(_) for _ in line.split(delim)] for line in txt.strip().split('\n')]
                        for line in mat:
                            assert len(line) == nreg
                        assert len(mat) == nreg
                        w = np.array(mat)
                        assert w.shape == (
                            nreg, nreg), f'wrong shape {w.shape}!=({nreg},{nreg})'
                        ws[i] = w[ti, tj]
                        ws[i] -= ws[i].min()
                        ws[i] /= np.percentile(ws[i], 99)
                        ws[i] = np.clip(ws[i], 0., 1.)
                        ws[i] = np.sqrt(ws[i])
                u = ws.mean(axis=0)
                ws -= u
                means.append(u)
                conns.append(ws)
        return parcs, means, conns

    @classmethod
    def from_old_pkl(cls, fname='xencode.pkl'):
        self = cls()
        with open(fname, 'rb') as fd:
            p = pickle.load(fd)
        self.wbs = p['all_wb1']
        self.conns = p['conns']
        self.means = p['means']
        self.parcs = p['parcs']
        self.tts = p['tts']
        return self

    @classmethod
    def combine_xc(cls, xc1, xc2, shuffle=True):
        """Combine cohorts loaded as xc1 & xc2. Used to load the
        HCP & 1KB datasets into a single dataset.
        """
        import jax, jax.numpy as jp
        xch = cls()
        n1, n2 = xc1.conns[0].shape[0], xc2.conns[0].shape[0]
        nh = n1 + n2
        xch.conns = [jp.concat([_1, _2]) for _1, _2 in zip(xc1.conns, xc2.conns)]
        # shuffle connectomes for training
        xch.permkey = jax.random.PRNGKey(42)
        xch.permidx = jax.random.permutation(xch.permkey, jp.r_[:nh],
                                             independent=True)
        for i in range(len(xch.conns)):
            xch.conns[i] = xch.conns[i][xch.permidx]
        # combine the rest
        xch.means = [(_1*n2 + _2*n2)/nh for _1, _2 in zip(xc1.means, xc2.means)]
        xch.parcs = xc1.parcs
        xch.tts = nh // 2
        xch.wbs = []
        return xch

    def to_pkl(self, fname='xcode.pkl'):
        stuff = dict(
            conns=self.conns,
            means=self.means,
            parcs=self.parcs,
            tts=self.tts,
            wbs=self.wbs
        )
        with open(fname, 'wb') as fd:
            pickle.dump(stuff, fd)

    @classmethod
    def from_pkl(cls, fname='xcode.pkl'):
        with open(fname, 'rb') as fd:
            stuff = pickle.load(fd)
        self = cls()
        for key, val in stuff.items():
            setattr(self, key, val)
        return self

    # these would be better on separate class
    def make_wbs(self, nlat):
        wbs = []
        for c in self.conns:
            n = c.shape[1]
            w1, w2t = small(2, n, nlat)
            b1 = small(nlat)
            b2 = small(n)
            wb = (w1, b1), (w2t.T, b2)  # enc, dec
            wbs.append(wb)
        return wbs

    def make_loss(self):
        import jax, jax.numpy as jp
        def loss(wbs, conns):
            ll = 0
            for ((ew, eb), _), c_e in zip(wbs, conns):
                u = c_e @ ew + eb  # encode from parc i
                for (_, (dw, db)), c_d in zip(wbs, conns):
                    v = u @ dw + db  # decode to parc j
                    ll = ll + jp.mean((v - c_d)**2)
            return ll
        loss = jax.jit(loss)
        grad = jax.jit(jax.grad(loss))
        return loss, grad

    # mini batch could be more effective, lower mem requirements
    def train(self, nlat, lr=3e-4, niter=500, nlog=None, tts=None, mb=64):
        import jax, jax.numpy as jp
        from jax.example_libraries import optimizers
        tts = tts or self.tts
        mbkey = jax.random.PRNGKey(mb)
        train_conns = [_[:tts] for _ in self.conns]
        test_conns = [_[tts:] for _ in self.conns]
        trace = []
        opt_init, opt_update, get_params = optimizers.adam(lr)
        wbs = self.make_wbs(nlat)
        opt_state = opt_init(wbs)
        nlog = nlog or niter
        loss, grad = self.make_loss()
        for i in (pbar := tqdm.trange(niter+1)):
            mbkey, _key = jax.random.split(mbkey)
            imb = jax.random.randint(mbkey, (mb,), 0, tts)
            mb_conns = [_[imb] for _ in train_conns]
            ll_train = np.log(loss(wbs, mb_conns))
            ll_test = np.log(loss(wbs, test_conns))
            trace.append((ll_train, ll_test))
            pbar.set_description(f'-ll {trace[0][1] - ll_test:0.3f}')
            wbs = get_params(opt_state)
            opt_state = opt_update(i, grad(wbs, mb_conns), opt_state)
        crs_test = all_conf_rates(wbs, test_conns).mean()
        self.wbs.append(wbs)
        return trace, wbs, crs_test

    @property
    def arch(self):
        # all_wb1[arch][conn][e,d][w,b]
        arch = [b.size for (((_, b), _), *_) in self.wbs]
        return arch

    def calc_mvn(self, arch, tts=None):
        import jax.numpy as jp
        iarch = self.arch.index(arch)
        tts = tts or self.tts
        us = []
        for ((ew, eb), _), c in zip(self.wbs[iarch], self.conns):
            us.append(c[tts:] @ ew + eb)
        us = jp.array(us)
        us_ = us.reshape(-1, us.shape[-1])
        u_mu = us_.mean(axis=0)
        u_cov = jp.cov(us_.T)
        return MvNorm(us, u_mu, u_cov)

    def get_triu(self, parc, tts=None):
        tts = tts or self.tts
        iparc = self.parcs.index(parc)
        return self.conns[iparc][self.tts:]

    def get_conn(self, parc, tts=None):
        iparc = self.parcs.index(parc)
        c_ = self.get_triu(parc, tts) + self.means[iparc]
        return triu_to_mat(c_)

    def decode_conn(self, parc, us):
        iarch = self.arch.index(us.shape[1])
        iparc = self.parcs.index(parc)
        _, (w, b) = self.wbs[iarch][iparc]
        mean = self.means[iparc]
        c_ = us @ w + b + mean
        return triu_to_mat(c_)

    def encode_conn(self, arch, parc, tts=None):
        iarch = self.arch.index(arch)
        iparc = self.parcs.index(parc)
        c_ = self.get_triu(parc, tts)
        (w, b), _ = self.wbs[iarch][iparc]
        return c_ @ w + b


def run_sbi(theta, features, fname=None, prog=True):
    import io, contextlib
    import torch
    from sbi.inference import NPE_A as NPE
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mu = to_torch(np.mean(theta, axis=0))
        cov = to_torch(np.cov(theta.T))
        prior = torch.distributions.MultivariateNormal(mu, cov)
        inference = NPE(prior=prior, show_progress_bars=prog)
        inference.append_simulations(to_torch(theta), to_torch(features))
        inference.train()
        posterior = inference.build_posterior()
    if prog:
        print(out.getvalue())
    if fname:
        with open(fname, 'wb') as fd:
            pickle.dump(posterior, fd)
    return posterior


def uniform_var(a,b):
    return (b - a)**2 / 12.


def posterior_diags(p_us, po_us, true_us):
    po_u = np.mean(po_us.numpy(), axis=0)
    po_sd = np.std(po_us.numpy(), axis=0)
    po_z = np.abs((po_u - true_us)/po_sd)
    p_var = np.var(p_us, axis=0) if hasattr(p_us, 'size') else uniform_var(*p_us)
    po_shrink = np.array(1 - po_sd**2/p_var)
    # check true in 90% ci
    q5, q95 = np.quantile(po_us, [0.05, 0.95], axis=0)
    ci90 = np.array((q5 < true_us) * (true_us < q95))
    return po_shrink, po_z, ci90


def mpr_dfun(ys, p):
    import vbjax as vb
    return vb.mpr_dfun(ys, (p[0]*p[-1]@ys[0], 0), vb.mpr_default_theta)


def hopf_dfun(ys, p):
    import vbjax as vb, jax.numpy as jp
    y0, y1 = ys
    cfun = vb.make_diff_cfun(jp.array(p[-1]))
    Ic = cfun(y0), # cfun(y1)
    # dy0 = y0 (eta - y0^2 - y1^2) - omega*y1
    return jp.array([y0 * (p[2]-y0**2-y1**2) - p[3]*y1 + 100.*p[0]*Ic[0],
                     y1 * (p[2]-y0**2-y1**2) + p[3]*y0  # + 100.*p[0]*Ic[1]
                 ])


class DynaModel:

    def __init__(self, name, dfun, features, dt=1e-3, adhoc=None,
                 key=None):
        import vbjax as vb, jax
        self.name = name
        self.dfun = hopf_dfun if self.name == 'hopf' else mpr_dfun
        self.dt = dt
        self.g = lambda x, p: p[1]
        _, loop = vb.make_sde(self.dt, self.dfun, self.g, adhoc=adhoc)
        self.loop = loop
        self.features = features
        self.key = key or jax.random.PRNGKey(42)

    def run_w(self, w, k, D, nwin, key, eta_mu=1.0, omega_scl=1.0):
        import jax, jax.numpy as jp, vbjax as vb
        w = w / w.max()
        if self.name == 'hopf':  # lame but
            eta = eta_mu + 0.1*vb.random.normal(self.key, shape=(w.shape[1],))
            omega = 2.*np.pi*jax.random.uniform(
                self.key, shape=(w.shape[1],),
                minval=0.9, maxval=1.1)*omega_scl
            p = k, D, eta, omega, w
        else:
            p = k, D, w

        def win(x0, key):
            z = vb.randn(1000, 2, w.shape[0], key=key)
            x = self.loop(x0, z, p)
            return x[-1], self.features(x)
        x0 = jp.zeros((2, w.shape[0])) + jp.c_[0., 0.].T+1e-4
        keys = jax.random.split(key, nwin)
        x0, xf = jax.lax.scan(win, x0, keys)
        return xf

    def run_ws(self, w, k, D, nwin=10,
               key=None, use_pmap=True):
        import jax, jax.numpy as jp, vbjax as vb
        key = key if key is None else jax.random.PRNGKey(42)
        def f(w, key, k, D): return self.run_w(w, k, D, nwin, key)
        if use_pmap:
            w_ = w.reshape((vb.cores, -1,) + w.shape[1:])
            keys_ = jax.random.split(key, w_.shape[:2])
            k_ = jp.array(k).reshape(w_.shape[:2])
            D_ = jp.array(D).reshape(w_.shape[:2])
            xf = jax.pmap(jax.vmap(f))(w_, keys_, k_, D_)
            xf = xf.reshape((-1,) + xf.shape[2:])
        else:
            keys = jax.random.split(key, w.shape[0])
            xf = jax.jit(jax.vmap(f))(w, keys, k, D)
        return xf[:, nwin//2:].mean(axis=1)


def make_dynamics(name: str, features=None, key=None):
    import jax, jax.numpy as jp, vbjax as vb
    key = key or jax.random.PRNGKey(42)
    # call make_dyamics('hopf') then pass it as "model" arg to other functions
    if name == 'mpr':
        if features is None:
            def features(x): return x[:, 1].mean(axis=0)
        model = DynaModel('mpr', mpr_dfun, features,
                          adhoc=vb.mpr_r_positive, key=key)
    elif name == 'hopf':
        if features is None:
            def features(x): return x[:, 0].var(axis=0)
        model = DynaModel('hopf', hopf_dfun, features, key=key)
    else:
        raise ValueError('model not implemented')
    return model.run_ws


# hopf without averaging and applies bold
# TODO : merge with hopf
def make_bold_hopf(features=None, key=None, with_bold=True):
    import jax, jax.numpy as jp, vbjax as vb
    key = key or jax.random.PRNGKey(42)

    features = features or (lambda x: x[:, 0].var(axis=0))
    hyper_key = key

    def hopf_dfun(ys, p):
        y0, y1 = ys
        cfun = vb.make_diff_cfun(jp.array(p[4]))
        Ic = cfun(y0), cfun(y1)
        return jp.array([y0 * (p[0]-y0**2-y1**2) - p[1]*y1 + p[2]*Ic[0],
                         y1 * (p[0]-y0**2-y1**2) + p[1]*y0 + p[2]*Ic[1]])

    def g(x, p): return p[3]
    dt = 1e-3  # needed for num stability
    _, loop = vb.make_sde(dt, hopf_dfun, g)

    def run_w(w, key, k=0.15, D=4e-1):
        w = w / w.max()
        # eta and omega are not inferred but have to be != for each node
        eta = -1. + vb.random.normal(hyper_key, shape=(w.shape[1],))
        omega = 2.*np.pi*jax.random.uniform(hyper_key, shape=(w.shape[1],),
                                            minval=0.02, maxval=0.04)
        p = eta, omega, 100.*k, D, w
        n = 1000
        if with_bold:
            n *= 100
            z = vb.randn(n, 2, w.shape[0], key=key)
            x0 = jp.zeros((2, w.shape[0])) + jp.c_[0., 0.].T
            x = loop(x0, z, p)
            bold_buf, bold_step, bold_sample = vb.make_bold(shape=(w.shape[0],),
                                                            dt=dt,
                                                            p=vb.bold_default_theta)
            bold_sample = vb.make_offline(
                step_fn=bold_step, sample_fn=bold_sample)
            windowed_r = x[:, 0].reshape(
                (-1, 200, w.shape[0]))  # len(bold)=500
            bold_buf, bold = jax.lax.scan(bold_sample, bold_buf, windowed_r)
            xf = features(bold)
        else:
            z = vb.randn(n, 2, w.shape[0], key=key)
            x0 = jp.zeros((2, w.shape[0])) + jp.c_[0., 0.].T
            x = loop(x0, z, p)
            xf = features(x)

        return xf

    def run_ws(w, k, D, key=hyper_key, use_pmap=True):
        def f(w, key, k, D): return run_w(w, key, k, D)
        if use_pmap:
            w_ = w.reshape((vb.cores, -1,) + w.shape[1:])
            keys_ = jax.random.split(key, w_.shape[:2])
            k_ = k.reshape(w_.shape[:2])
            D_ = D.reshape(w_.shape[:2])
            xf = jax.pmap(jax.vmap(f))(w_, keys_, k_, D_)
            xf = xf.reshape((-1,) + xf.shape[2:])
        else:
            keys = jax.random.split(key, w.shape[0])
            xf = jax.jit(jax.vmap(f))(w, keys, k, D)
        return xf

    return run_ws


def sample_subj_model(w, model, num_batch, batch_size, use_pmap=True, prog=True):
    import jax, jax.numpy as jp, vbjax as vb
    w = w + jp.zeros((batch_size, 1, 1))
    assert w.ndim == 3
    parm_keys = jax.random.split(vb.key, (2, num_batch))
    thetas, xfs = [], []
    iters = (tqdm.trange if prog else range)(num_batch)
    for i in iters:
        # TODO generalize priors
        k = 0.1 + vb.rand(batch_size, key=parm_keys[0, i])*0.2
        D = 0.2 + vb.rand(batch_size, key=parm_keys[1, i])*0.2
        xf = model(w, k, D, use_pmap=use_pmap)
        thetas.append(jp.c_[k, D])
        xfs.append(xf)
    thetas = jp.array(thetas).reshape(-1, thetas[0].shape[1])
    xfs = jp.array(xfs).reshape(-1, xfs[0].shape[1])
    return thetas, xfs


def sample_model(xc, model, mvn, parc, num_batch, batch_size, prog=True, use_pmap=True):
    import jax, jax.numpy as jp, vbjax as vb

    parm_keys = jax.random.split(vb.key, (2, num_batch))
    thetas, xfs = [], []
    iters = (tqdm.trange if prog else range)(num_batch)
    for i in iters:
        # TODO generalize priors
        u = mvn.sample(batch_size)
        k = 0.1 + vb.rand(batch_size, key=parm_keys[0, i])*0.2
        D = 0.2 + vb.rand(batch_size, key=parm_keys[1, i])*0.2
        w = xc.decode_conn(parc, u)
        xf = model(w, k, D, use_pmap=use_pmap)
        thetas.append(jp.concat([u, jp.vstack([k, D]).T], axis=1))
        xfs.append(xf)
    thetas = jp.array(thetas).reshape(-1, thetas[0].shape[1])
    xfs = jp.array(xfs).reshape(-1, xfs[0].shape[1])
    return thetas, xfs


# in-sample
def bench_cohort_model(
    xc: XCode,
    model: Callable,
    parc: str = '079-Shen2013',
    arch: int = 8,
    num_batch: int = 32,
    batch_size: int = 128,
    use_pmap=True
):

    mvn = xc.calc_mvn(arch)
    thetas, xfs = sample_model(
        xc, model, mvn, parc, num_batch, batch_size,
        use_pmap=use_pmap)
    posterior = run_sbi(thetas, xfs)
    thetas_hat = posterior.sample_batched((200,), x=to_torch(xfs[:batch_size]))
    ps, pz, ci = posterior_diags(thetas, thetas_hat, thetas[:batch_size])
    return ps.mean().item(), pz.mean()


# TODO finish up..
def bench_model(xc, model, parc='079-Shen2013',
                arch=8, num_batch=32, batch_size=128,
                do_subjets=False, num_postcd=None, inflate=1,
                use_pmap=True, return_everything=False,
                prog=True
                ):

    import jax, jax.numpy as jp, vbjax as vb

    # setup ground truth
    w = xc.get_conn(parc)  # (n_test, n_parc, n_parc)
    u = xc.encode_conn(arch, parc)
    k = 0.2 + vb.rand(len(w))*0.2
    D = 0.2 + vb.rand(len(w))*0.2
    theta = jp.concat([u, jp.c_[k, D]], axis=1)
    xf = model(w, k, D, use_pmap=False)  # (n_test, n_parc)

    # build & apply cohort sbi
    mvn = xc.calc_mvn(arch)
    mvn.u_cov = mvn.u_cov * inflate  # inflate a bit
    theta_cdhat, xf_cdhat = sample_model(xc, model, mvn, parc, num_batch, batch_size,
                                         prog=prog, use_pmap=use_pmap)
    posterior_cd = run_sbi(theta_cdhat, xf_cdhat, prog=prog)
    po_theta_cdhat = posterior_cd.sample_batched(
        (num_postcd or theta_cdhat.shape[0],), x=np.array(xf), show_progress_bars=prog)
    # (nsamp, xf.shape[0], arch+2)
    diags_cd = posterior_diags(
        theta_cdhat[:, arch:], po_theta_cdhat[..., arch:], theta[:, arch:])

    if do_subjets:
        # sbi subject
        diags_subj = []
        for it in tqdm.trange(len(xf)):  # 74 subjects
            theta_hat, xf_hat = sample_subj_model(
                w[it], model, num_batch, batch_size, prog=prog, use_pmap=use_pmap)
            posterior = run_sbi(theta_hat, xf_hat, prog=prog)
            po_theta_hat = posterior.sample(
                (theta_hat.shape[0],), x=np.array(xf[it]), show_progress_bars=prog)
            diags_it = posterior_diags(
                theta_hat, po_theta_hat, theta[it, arch:])
            diags_subj.append(diags_it)
    else:
        diags_subj = None

    if return_everything:
        return locals()
    # TODO compare stats of diags from sbi-subject to sbi-cohort
    return diags_cd, diags_subj


class DataArgs(tap.TypedArgs):
    input: str = tap.arg(help="input fname")
    pkl: str = tap.arg(help="output pkl fname")
    hcp: bool = tap.arg(default=False, help='is hcp dataset?')
    combine: bool = tap.arg(default=False, help='combine datasets')
    skip_parc: str = tap.arg(default='', help='parcellations to skip')


class TrainArgs(tap.TypedArgs):
    data: str = tap.arg(help="data to use: hcp, 1kb or both")
    arch: int = tap.arg(default=8, help="Architecture: latent dimension")
    niter: int = tap.arg(default=100, help="Number of optimization iterations")
    out: str = tap.arg(help="Output filename")
    mb: int = tap.arg(default=64, help="mini-batch size")


def run_data(args: DataArgs) -> None:
    print(args)
    if args.combine:
        pkl_hcp, pkl_1kb = args.input.split(',')
        xc_hcp = XCode.from_pkl(pkl_hcp)
        xc_1kb = XCode.from_pkl(pkl_1kb)
        XCode.combine_xc(xc_hcp, xc_1kb).to_pkl(args.pkl)
    elif args.input and args.pkl:
        XCode.from_kg(args.input, hcp=args.hcp).to_pkl(args.pkl)
    else:
        print('unhandled case', args)


def run_train(args: TrainArgs) -> None:
    print(args)
    xc: XCode = XCode.from_pkl(args.data)
    trace, wbs, crs_test = xc.train(args.arch, niter=args.niter)
    print('trained confusion rate: ', crs_test)
    import pylab as pl
    pl.plot(trace)
    pl.savefig(f'{args.out}.pdf')
    pl.close()
    xc.to_pkl(args.out)

class HopfTestArgs(tap.TypedArgs):
    pass

def run_hopf_test(args: HopfTestArgs) -> None:
    """Here we just test that the simulation is relatively
    identifying for the connectome in a particular regime. 
    """
    import pylab as pl
    import aaicd
    import jax, jax.numpy as jp, vbjax as vb

    xc = aaicd.XCode.from_pkl('both.pkl')
    parc = '079-Shen2013'
    arch, *_ = xc.arch

    # setup ground truth
    w = xc.get_conn(parc)  # (n_test, n_parc, n_parc)
    w = w[:8]
    assert w.shape == (8, 79, 79)
    u = xc.encode_conn(arch, parc)
    u = u[:8]
    k = 0.2 + vb.rand(len(w))*0.2
    D = 0.2 + vb.rand(len(w))*0.2
    theta = jp.concat([u, jp.c_[k, D]], axis=1)

    key = vb.keys[0]
    k = 0.01
    D = 0.1
    dt = 0.02

    # run sim w/ fc as feature
    ti, tj = jp.triu_indices(79, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]

    model = aaicd.DynaModel('hopf', aaicd.hopf_dfun, features, dt=dt)
    fcs = []
    for i in range(8):
        fc1 = model.run_w(w[i], k, D, nwin=10, key=vb.keys[0]).mean(axis=0)
        fc2 = model.run_w(w[i], k, D, nwin=10, key=vb.keys[1]).mean(axis=0)
        assert fc1.shape == ti.shape
        fcs.append((fc1, fc2))

    sim = np.zeros((8,8))
    for i in range(8):
        fi = fcs[i][0].reshape(-1)
        for j in range(8):
            fj = fcs[j][1].reshape(-1)
            sim[i, j] = np.sum(np.square(fi - fj))

    np.testing.assert_array_equal(
        np.argmin(sim, axis=1), np.r_[:len(w)])
    # pl.imshow(-sim)
    # pl.show()

class HopfSampleArgs(tap.TypedArgs):
    data: str = tap.arg('-d', default='both.pkl')
    seed: int = tap.arg('-s', default=42)
    parc: str = tap.arg('-p', default='079-Shen2013')
    arch: int = tap.arg('-a', default=10)
    num_batch: int = tap.arg('-n', default=1)
    batch_size: int = tap.arg('-b', default=8)
    use_pmap: bool = tap.arg(default=False, help='use all cores')
    out_npz: str = tap.arg('-o')
    per_subj: bool = tap.arg(default=False, help='per subj sampling')


def run_hopf_sample(args: HopfSampleArgs) -> None:
    print(args)
    import pylab as pl
    import jax, jax.numpy as jp, vbjax as vb

    # load data
    xc = XCode.from_pkl(args.data)
    nreg = int(args.parc.split('-')[0])
    iparc = xc.parcs.index(args.parc)
    test_ws = triu_to_mat(xc.conns[iparc][xc.tts:] + xc.means[iparc])

    # simulation & features
    ti, tj = jp.triu_indices(nreg, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]
    model = DynaModel('hopf', hopf_dfun, features, dt=0.02)
    def f(w, k, D, key):
        return model.run_w(w, k, D, nwin=10, key=key).mean(axis=0)
    f = jax.vmap(f)
    if args.use_pmap:
        f = jax.pmap(f)
    else:
        f = jax.jit(f)

    # data, parameters
    B, A = args.batch_size, args.arch
    if A not in xc.arch:
        A, *_ = xc.arch
    mvn = xc.calc_mvn(A)
    keys = jax.random.split(
        jax.random.PRNGKey(args.seed), (args.num_batch, 3 + B))
    if args.per_subj:
        theta = np.zeros((args.num_batch, B, 2), 'f')
    else:
        theta = np.zeros((args.num_batch, B, A + 2), 'f')
    feats = np.zeros((args.num_batch, B, ti.size), 'f')
    i_test_ws = np.zeros((args.num_batch, B), dtype=np.uint16)
    for i_batch in tqdm.trange(args.num_batch):
        key = keys[i_batch]
        if args.per_subj: # sample test ws
            i_test_ws[i_batch] = jax.random.randint(
                key[0], (B, ), minval=0, maxval=test_ws.shape[0])
            w = test_ws[i_test_ws[i_batch]]
        else: # sampling cohort prior over u
            u = mvn.sample(B)
            w = xc.decode_conn(args.parc, u)
        # k = 0.01 + vb.rand(B, key=key[1]) * 0.01
        # D = 0.1 + vb.rand(B, key=key[2]) * 0.1
        lk = jax.random.normal(key[1], (B, )) - 6.0
        lD = jax.random.normal(key[2], (B, )) - 1.5
        k = jp.exp(lk)
        D = jp.exp(lD)
        if args.use_pmap:
            # reshape leading
            rl = lambda a: a.reshape((vb.cores, -1) + a.shape[1:])
            fc = f(*(rl(_) for _ in (w, k, D, key[3:])))
            fc = fc.reshape((B,) + fc.shape[2:])
        else:
            fc = f(w, k, D, key[3:])
        fc.block_until_ready()
        # save results
        if args.per_subj:
            theta[i_batch, :, 0] = lk
            theta[i_batch, :, 1] = lD
        else:
            theta[i_batch, :, :A] = u
            theta[i_batch, :, A] = lk
            theta[i_batch, :, A+1] = D
        feats[i_batch] = fc

    print(f'saving to {args.out_npz}')
    outputs = dict(theta=theta, feats=feats, arch=A, parc=args.parc)
    if args.per_subj:
        outputs['i_test_ws'] = i_test_ws
        outputs['test_ws'] = test_ws
    np.savez(args.out_npz, **outputs)


class SBIArgs(tap.TypedArgs):
    samples: str = tap.arg('-s', help='samples file')
    sbi_pkl: str = tap.arg('-o', help='pickle file to store sbi posterior')


def run_sbi_args(args: SBIArgs):
    npz = np.load(args.samples, allow_pickle=True)
    theta = npz['theta']  # u, k, D
    feats = npz['feats']  # fc
    theta = theta.reshape(-1, theta.shape[-1])
    feats = feats.reshape(-1, feats.shape[-1])
    assert theta.shape[0] == feats.shape[0]
    per_subj = 'i_test_ws' in npz
    if per_subj:
        print(f'per subject SBI w/ theta {theta.shape}, features {feats.shape}; saving to {args.sbi_pkl}')
        # run_sbi(theta, feats, fname=args.sbi_pkl)
        iw = npz['i_test_ws'].reshape(-1)
        tw = npz['test_ws']
        uiw = np.unique(iw)
        print(iw.shape, tw.shape, feats.shape, theta.shape)
        # (32768,) (231, 79, 79) (32768, 3081) (32768, 2)
        subj_posts = []
        for subj in tqdm.tqdm(uiw):
            subj_mask = iw == subj
            subj_post = run_sbi(theta[subj_mask], feats[subj_mask], prog=False)
            subj_posts.append(subj_post)
        with open(args.sbi_pkl, 'wb') as fd:
            pickle.dump(subj_posts, fd)
    else:
        print(f'cohort SBI w/ theta {theta.shape}, features {feats.shape}; saving to {args.sbi_pkl}')
        run_sbi(theta, feats, fname=args.sbi_pkl)


class EvalSBIArgs(tap.TypedArgs):
    pass


def run_eval(args: EvalSBIArgs):
    import sbi
    from sbi.inference.posteriors import DirectPosterior
    from typing import List
    # read files, names in run.sh TODO args
    cohort_samp = np.load('cohort-samples.npz', allow_pickle=True)
    cohort_post: DirectPosterior = load_pkl('cohort-posterior.pkl')
    subj_samp = np.load('subj-samples.npz', allow_pickle=True)
    subj_post: List[DirectPosterior] = load_pkl('subj-posterior.pkl')
    # extract relevant arrays
    iw = subj_samp['i_test_ws'].reshape(-1)
    tw = subj_samp['test_ws']
    uf = lambda a: a.reshape((-1,) + a.shape[2:])
    s_theta = uf(subj_samp['theta'])
    s_feats = uf(subj_samp['feats'])
    c_theta = uf(cohort_samp['theta'])
    c_feats = uf(cohort_samp['feats'])
    arch = cohort_samp['arch']
    # go over each subject, apply cohort posterior to the subject
    # and make two comparisons:
    # 1. is the subject connectome recovered?
    # 2. how well the per-subject parameters recovered?
    for subj in np.unique(iw): #tqdm.tqdm(np.unique(iw)):
        mask = iw == subj
        _sample = lambda post: post.sample_batched(
            (200,), to_torch(s_feats[mask]), show_progress_bars=False)
        try:
            cp_theta = _sample(cohort_post)
            sp_theta = _sample(subj_post[subj])  # (200, 75, arch + 2)
        except RuntimeError:
            print(subj, 'fail')
            continue
        import pylab as pl
        # subject sbi
        s, sz, sci90 = posterior_diags(p_us=s_theta, po_us=sp_theta, true_us=s_theta[mask])
        s, cz, cci90 = posterior_diags(
            p_us=s_theta, po_us=cp_theta[..., arch:], true_us=s_theta[mask])
        print(subj, mask.sum(), 'z', sz.mean(), np.quantile(cz, 0.5),
              sci90.mean()*100, cci90.mean()*100, "% ok")

        pl.figure()
        for i in range(25):
            pl.subplot(5, 5, i + 1)
            pl.hist(s_theta[:, 0], alpha=0.2, density=True, label='prior', log=True)
            pl.hist(cp_theta[:, i, arch], alpha=0.5, density=True, label='cohort', log=True)
            pl.hist(sp_theta[:, i, 0], alpha=0.5, density=True, label='subject', log=True)
            pl.axvline(s_theta[mask][i, 0], color='r', label='true')
            # pl.legend()
        pl.tight_layout()
        pl.show()
        1/0


class DownloadDataArgs(tap.TypedArgs):
    pass


def run_dl(args: DownloadDataArgs):
    XCode._download_kg_zip('hcp.zip', hcp=True)
    XCode._download_kg_zip('1kb.zip', hcp=False)


class DebugArgs(tap.TypedArgs):
    show_crs: bool = tap.arg(default=False)

def run_debug(args: DebugArgs):
    print('\n\n' + 'DEBUG ' * 10)
    print(args)
    import pylab as pl
    import jax, jax.numpy as jp, vbjax as vb
    import sbi

    # load data
    xc = XCode.from_pkl('both.pkl')

    if args.show_crs:
        c = [_[xc.tts:] for _ in xc.conns]
        crs = all_conf_rates(xc.wbs[0], c)
        pl.imshow(crs, vmin=0, vmax=1.0); pl.colorbar()
        pl.show()

    parc = '079-Shen2013'
    iparc = xc.parcs.index(parc)
    arch = A = 16
    assert arch in xc.arch
    nreg = int(parc.split('-')[0])

    ti, tj = jp.triu_indices(nreg, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]
    model = DynaModel('hopf', hopf_dfun, features, dt=0.02)
    def f(w, k, D, key):
        return model.run_w(w, k, D, nwin=10, key=key).mean(axis=0)
    
    mvn = xc.calc_mvn(A)
    ng = 64
    k, D = jp.exp(jp.mgrid[-8:-4:1j*ng, -2:1:1j*ng])
    k, D = k.reshape(-1), D.reshape(-1)
    print('k', k.min(), k.max())
    print('D', D.min(), D.max())
    u = mvn.sample(k.size)
    w = xc.decode_conn(parc, u)
    keys = jax.random.split(vb.key, k.size)
    x = apply(f, w, k, D, keys, B=ng)
    print(x.shape)

    # post = load_pkl('cohort-posterior.pkl')
    lk, lD = np.log(k), np.log(D)
    samp = np.c_[u, lk, lD]
    post = run_sbi(samp, x)
    theta_hat = post.sample_batched((200, ), to_torch(x))
    print(theta_hat.shape)  # (200, 64, 18)

    s, z, ok90 = posterior_diags(samp, theta_hat, samp)
    print(s.shape)

    pl.figure()
    for i in range(s.shape[-1]):
        pl.subplot(5, 4, i + 1)
        s, z, ok90 = posterior_diags(samp[:, i], theta_hat[:, :, i], samp[:, i])
        pl.plot(s, z, 'x')
        print(i, ok90.mean())

    pl.figure()
    # i_par = 0
    for i_par in range(18):
        pl.subplot(5, 4, i_par + 1)
        i_samp = 0
        pl.hist(samp[:, i_par], alpha=0.5)
        pl.hist(theta_hat[:, i_samp, i_par], alpha=0.5)
        pl.axvline(samp[i_samp, i_par], color='r')

    pl.show()

    print('\n'*3, 'DONE '*10)


def apply(f, *args, B=1):
    import vbjax as vb, jax, jax.numpy as jp
    args_ = [_.reshape((-1, vb.cores, B//vb.cores) + _.shape[1:]) for _ in args]
    pvf = jax.pmap(jax.vmap(f))
    xs = jp.array([
        pvf(*[_[i] for _ in args_])
        for i in range(args_[0].shape[0])])
    return xs.reshape((-1, ) + xs.shape[3:])


if __name__ == '__main__':
    # this is stupid TODO switch to https://github.com/google/python-fire
    tap.Parser(
        tap.SubParserGroup(
            tap.SubParser('data', DataArgs, help="help of data"),
            tap.SubParser('train', TrainArgs, help="help of train"),
            tap.SubParser('hopf_test', HopfTestArgs, help="help of train"),
            tap.SubParser('hopf_sample', HopfSampleArgs, help="sample hopf model"),
            tap.SubParser('sbi', SBIArgs, help="run sbi"),
            tap.SubParser('eval', EvalSBIArgs, ),
            tap.SubParser('download', DownloadDataArgs,
                          help="Download datasets from KG."),
            tap.SubParser('dbg', DebugArgs),
        )
    ).bind(
        run_data,
        run_train,
        run_hopf_test,
        run_hopf_sample,
        run_sbi_args,
        run_eval,
        run_dl,
        run_debug,
    ).run()

