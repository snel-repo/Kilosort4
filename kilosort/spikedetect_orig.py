import logging
from io import StringIO

logger = logging.getLogger(__name__)

from pdb import set_trace

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from kilosort.utils import template_path
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from torch.nn.functional import avg_pool2d, conv1d, max_pool1d, max_pool2d
from tqdm import tqdm

device = torch.device("cuda")


def my_max2d(X, dt):
    Xmax = max_pool2d(
        X.unsqueeze(0),
        [2 * dt[0] + 1, 2 * dt[1] + 1],
        stride=[1, 1],
        padding=[dt[0], dt[1]],
    )
    return Xmax[0]


def my_sum2d(X, dt):
    Xsum = avg_pool2d(
        X.unsqueeze(0),
        [2 * dt[0] + 1, 2 * dt[1] + 1],
        stride=[1, 1],
        padding=[dt[0], dt[1]],
    )
    Xsum *= (2 * dt[0] + 1) * (2 * dt[1] + 1)
    return Xsum[0]


def extract_snippets(
    X,
    nt,
    twav_min,
    Th_single_ch,
    loc_range=[4, 5],
    long_range=[6, 30],
    device=torch.device("cuda"),
):
    Xabs = X.abs()
    Xmax = my_max2d(Xabs, loc_range)
    ispeak = torch.logical_and(Xmax == Xabs, Xabs > Th_single_ch[0]).float()

    ispeak_sum = my_sum2d(ispeak, long_range)
    is_peak_iso = (ispeak_sum == 1) * (ispeak == 1)

    is_peak_iso[:, :nt] = 0
    is_peak_iso[:, -nt:] = 0

    xy = is_peak_iso.nonzero()

    clips = X[xy[:, :1], xy[:, 1:2] - twav_min + torch.arange(nt, device=device)]
    return clips


def extract_wPCA_wTEMP(
    ops,
    bfile,
    nt=61,
    twav_min=20,
    Th_single_ch=6,
    nskip=25,
    device=torch.device("cuda"),
):

    clips = np.zeros((500000, nt), "float32")
    i = 0
    for j in range(0, bfile.n_batches, nskip):
        X = bfile.padded_batch_to_torch(j, ops)

        clips_new = extract_snippets(
            X, nt=nt, twav_min=twav_min, Th_single_ch=Th_single_ch, device=device
        )

        nnew = len(clips_new)

        if i + nnew > clips.shape[0]:
            break

        clips[i : i + nnew] = clips_new.cpu().numpy()
        i += nnew

    clips = clips[:i]
    clips /= (clips**2).sum(1, keepdims=True) ** 0.5

    model = TruncatedSVD(n_components=ops["settings"]["n_pcs"]).fit(clips)
    wPCA = torch.from_numpy(model.components_).to(device).float()

    model = KMeans(n_clusters=ops["settings"]["n_templates"], n_init=10).fit(clips)
    wTEMP = torch.from_numpy(model.cluster_centers_).to(device).float()
    wTEMP = wTEMP / (wTEMP**2).sum(1).unsqueeze(1) ** 0.5
    do_plot = False
    if do_plot:
        # make KMeans clustering plot
        # but we must first project the clips into the PCA space
        # also we need to project the cluster centers into the PCA space

        # define the colors for the clusters
        n_colors = ops["settings"]["n_templates"]
        colors = [
            f"hsl({h}, 50%, 50%)"
            for h in np.linspace(0, 360, n_colors, endpoint=False).astype(int)
        ]
        fig2 = go.Figure()
        proj_clips = clips @ wPCA.cpu().numpy().T.astype(float)
        proj_centers = (wTEMP @ wPCA.T).cpu().numpy().astype(float)
        for i in range(ops["settings"]["n_templates"]):
            cluster_points = proj_clips[model.labels_ == i]
            cluster_centers = proj_centers[i]
            fig2.add_trace(
                go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode="markers",
                    marker=dict(size=6, color=colors[i]),
                    name=f"Cluster {i}",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=[cluster_centers[0]],
                    y=[cluster_centers[1]],
                    mode="markers",
                    marker=dict(size=10, color=colors[i]),
                    name=f"Cluster {i} center",
                )
            )
        fig2.update_layout(
            title="Kmeans Clustering in 61D Space of Spikes, Projected into 2D PCA Space",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
        )
        fig2.show()
        # use plotly express to plot wTEMP all overlaid on each other
        # make sure to use the same colors as the clusters

        fig3 = px.line(
            wTEMP.cpu().numpy().T,
            color_discrete_sequence=colors,
            title="Single Channel Templates (Cluster Centers, Kmeans Clustered in 61D Space of Spikes)",
        )
        fig3.show()
        set_trace()
    return wPCA, wTEMP


def get_waves(ops, device=torch.device("cuda")):
    dd = np.load(template_path())
    wTEMP = torch.from_numpy(dd["wTEMP"]).to(device)
    wPCA = torch.from_numpy(dd["wPCA"]).to(device)
    return wPCA, wTEMP


def template_centers(ops):
    shank_idx = ops["kcoords"]
    xc = ops["xc"]
    yc = ops["yc"]
    dmin = ops["settings"]["dmin"]
    if dmin is None:
        # Try to determine a good value automatically based on contact positions.
        dmin = np.median(np.diff(np.unique(yc)))
    ops["dmin"] = dmin
    ops["dminx"] = dminx = ops["settings"]["dminx"]

    # Iteratively determine template placement for each shank separately.
    yup = np.array([])
    xup = np.array([])
    for i in np.unique(shank_idx):
        xc_i = xc[shank_idx == i]
        yc_i = yc[shank_idx == i]
        xmin, xmax, ymin, ymax = xc_i.min(), xc_i.max(), yc_i.min(), yc_i.max()

        yup = np.concatenate([yup, np.arange(ymin, ymax + 0.00001, dmin / 2)])
        nx = np.round((xmax - xmin) / (dminx / 2)) + 1
        xup = np.concatenate([xup, np.linspace(xmin, xmax, int(nx))])

    ops["yup"] = yup
    ops["xup"] = xup

    # Set max channel distance based on dmin, dminx, use whichever is greater.
    if ops.get("max_channel_distance", None) is None:
        ops["max_channel_distance"] = max(dmin, dminx)

    return ops


def template_match(X, ops, iC, iC2, weigh, device=torch.device("cuda")):
    NT = X.shape[-1]
    nt = ops["nt"]
    Nchan = ops["Nchan"]
    Nfilt = iC.shape[1]

    tch0 = torch.zeros(1, device=device)
    tch1 = torch.ones(1, device=device)

    W = ops["wTEMP"].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt // 2)

    nt0 = ops["settings"]["nt0min"]
    nk = ops["settings"]["n_templates"]

    niter = 40
    nb = (NT - 1) // niter + 1
    As = torch.zeros((Nfilt, NT), device=device)
    Amaxs = torch.zeros((Nfilt, NT), device=device)
    imaxs = torch.zeros((Nfilt, NT), dtype=torch.int64, device=device)

    ti = torch.arange(Nfilt, device=device)
    tj = torch.arange(nb, device=device)

    for t in range(niter):
        A = torch.einsum("ijk, jklm-> iklm", weigh, B[iC, :, nb * t : nb * (t + 1)])
        A = A.transpose(1, 2)
        A = A.reshape(-1, Nfilt, A.shape[-1])

        # Aa, imax = torch.max(A, 0)
        Aa, imax = torch.max(A.abs(), 0)
        imax = (1 + imax) * A[imax, ti.unsqueeze(-1), tj[: A.shape[-1]]].sign()

        As[:, nb * t : nb * (t + 1)] = Aa
        imaxs[:, nb * t : nb * (t + 1)] = imax
        Amax = torch.max(Aa[iC2], 0)[0]
        Amaxs[:, nb * t : nb * (t + 1)] = Amax

    Amaxs[:, :nt] = 0
    Amaxs[:, -nt:] = 0
    Amaxs = max_pool1d(
        Amaxs.unsqueeze(0), (2 * nt0 + 1), stride=1, padding=nt0
    ).squeeze(0)
    xy = torch.logical_and(Amaxs == As, As > ops["Th_universal"]).nonzero()
    imax = imaxs[xy[:, 0], xy[:, 1]]
    amp = As[xy[:, 0], xy[:, 1]]

    ssign = imax.sign()
    imax = imax.abs() - 1
    adist = B[iC[:, xy[:, 0]], imax % nk, xy[:, 1]] * ssign

    # adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]]

    # xy[:,1] -= nt
    return xy, imax, amp, adist


def nearest_chans(ys, yc, xs, xc, nC, device=torch.device("cuda")):
    ds = (ys - yc[:, np.newaxis]) ** 2 + (xs - xc[:, np.newaxis]) ** 2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(device)
    ds = np.sort(ds, 0)[:nC]

    return iC, ds


def yweighted(yc, iC, adist, xy, device=torch.device("cuda")):

    yy = torch.from_numpy(yc).to(device)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0 / cF0.sum(0)

    yct = (cF0 * yy[:, xy[:, 0]]).sum(0)
    return yct


def run(ops, bfile, device=torch.device("cuda"), progress_bar=None):
    sig = ops["settings"]["min_template_size"]
    nsizes = ops["settings"]["template_sizes"]

    if ops["settings"]["templates_from_data"]:
        logger.info("Re-computing universal templates from data.")
        # Determine templates and PC features from data.
        ops["wPCA"], ops["wTEMP"] = extract_wPCA_wTEMP(
            ops,
            bfile,
            nt=ops["nt"],
            twav_min=ops["nt0min"],
            Th_single_ch=ops["settings"]["Th_single_ch"],
            nskip=ops["settings"]["nskip"],
            device=device,
        )
        do_plot = False
        if do_plot:
            # plot these to show the templates in different rows using plotly
            # define the colors for the clusters
            n_colors = ops["settings"]["n_templates"]
            colors = [
                f"hsl({h}, 50%, 50%)"
                for h in np.linspace(0, 360, n_colors, endpoint=False).astype(int)
            ]
            print("PLOTTING NOW")
            pio.renderers.default = "browser"
            fig = make_subplots(
                rows=ops["settings"]["n_templates"],
                cols=1,
                shared_xaxes="all",
                shared_yaxes="all",
            )
            for i in range(ops["settings"]["n_templates"]):
                fig.add_trace(
                    go.Scatter(
                        y=ops["wTEMP"][i].cpu().numpy(),
                        mode="lines",
                        line=dict(color=colors[i]),
                    ),
                    row=i + 1,
                    col=1,
                )
            fig.update_layout(
                title="Single Channel Templates (Cluster Centers, Kmeans Clustered in 61D Space of Spikes)",
                xaxis_title="Time (samples)",
                yaxis_title="Amplitude",
            )
            fig.show()
            set_trace()
    else:
        logger.info("Using built-in universal templates.")
        # Use pre-computed templates.
        ops["wPCA"], ops["wTEMP"] = get_waves(ops, device=device)

    ops = template_centers(ops)
    [ys, xs] = np.meshgrid(ops["yup"], ops["xup"])
    ys, xs = ys.flatten(), xs.flatten()
    xc, yc = ops["xc"], ops["yc"]
    Nfilt = len(ys)

    nC = ops["settings"]["nearest_chans"]
    nC2 = ops["settings"]["nearest_templates"]
    iC, ds = nearest_chans(ys, yc, xs, xc, nC, device=device)

    # Don't use templates that are too far away from nearest channel
    # (use square of max distance since ds are squared distances)
    igood = ds[0, :] <= ops["max_channel_distance"] ** 2
    iC = iC[:, igood]
    ds = ds[:, igood]
    ys = ys[igood]
    xs = xs[igood]
    ops["ycup"], ops["xcup"] = ys, xs

    iC2, ds2 = nearest_chans(ys, ys, xs, xs, nC2, device=device)

    ds_torch = torch.from_numpy(ds).to(device).float()
    template_sizes = sig * (1 + torch.arange(nsizes, device=device))
    weigh = torch.exp(-ds_torch.unsqueeze(-1) / template_sizes**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1) ** 0.5

    st = np.zeros((10**6, 6), "float64")
    tF = np.zeros((10**6, nC, ops["settings"]["n_pcs"]), "float32")

    k = 0
    nt = ops["nt"]
    tarange = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
    s = StringIO()
    for ibatch in tqdm(
        np.arange(bfile.n_batches),
        miniters=200 if progress_bar else None,
        mininterval=60 if progress_bar else None,
    ):
        X = bfile.padded_batch_to_torch(ibatch, ops)

        xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=device)
        yct = yweighted(yc, iC, adist, xy, device=device)
        nsp = len(xy)

        if k + nsp > st.shape[0]:
            st = np.concatenate((st, np.zeros_like(st)), 0)
            tF = np.concatenate((tF, np.zeros_like(tF)), 0)

        xsub = X[iC[:, xy[:, :1]], xy[:, 1:2] + tarange]
        xfeat = xsub @ ops["wPCA"].T
        tF[k : k + nsp] = xfeat.transpose(0, 1).cpu().numpy()

        st[k : k + nsp, 0] = (
            ((xy[:, 1] - nt) / ops["fs"] + ibatch * (ops["batch_size"] / ops["fs"]))
            .cpu()
            .numpy()
        )
        st[k : k + nsp, 1] = yct.cpu().numpy()
        st[k : k + nsp, 2] = amp.cpu().numpy()
        st[k : k + nsp, 3] = imax.cpu().numpy()
        st[k : k + nsp, 4] = ibatch
        st[k : k + nsp, 5] = xy[:, 0].cpu().numpy()

        k = k + nsp

        if progress_bar is not None:
            progress_bar.emit(int((ibatch + 1) / bfile.n_batches * 100))

    st = st[:k]
    tF = tF[:k]
    ops["iC"] = iC
    ops["iC2"] = iC2
    ops["weigh"] = weigh
    return st, tF, ops
