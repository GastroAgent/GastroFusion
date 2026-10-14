import collections
import argparse
import pickle
import random
import os
import json
import math

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from numpy import linalg as LA

from tqdm import tqdm


# ============================================================
#                   默认实验参数
# ============================================================

DATASET = "medical"

# test-test:  support=test, query=test（同一 episode 图像不重叠）
# test-train: support=train, query=test
EVAL_MODE = "test-train"

# 当前测试数据
MEDICAL_SUBSET = "食管_8way_1query"

# 必须和 save_plk.py 使用的一致
MODEL_NAME = "ResNet18"
METHOD = "S2M2_R"

# Few-shot setting
N_WAYS = 8
N_SHOT = 5
N_QUERIES = 1
N_RUNS = 100

# PT-MAP parameters
LAM = 10.0
ALPHA = 0.2
N_ITER = 20
BETA = 0.5

# 结果输出
OUTPUT_JSON = (
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/tsy/pt_map/test_train/食管_ResNet18_S2M2_R_"
    "ptmap_8way_5shot_1query_100ep.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PT-MAP medical episodic evaluation"
    )
    parser.add_argument(
        "--eval_mode", choices=["test-test", "test-train"],
        default=EVAL_MODE
    )
    parser.add_argument("--medical_subset", default=MEDICAL_SUBSET)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--n_way", type=int, default=N_WAYS)
    parser.add_argument("--n_shot", type=int, default=N_SHOT)
    parser.add_argument("--n_query", type=int, default=N_QUERIES)
    parser.add_argument("--episodes", type=int, default=N_RUNS)
    parser.add_argument("--seed", type=int, default=12667)
    parser.add_argument("--lam", type=float, default=LAM)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--iterations", type=int, default=N_ITER)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument(
        "--feature_root", default="./checkpoints/medical",
        help="包含 <subset>/<model>_<method>/last 的目录"
    )
    parser.add_argument(
        "--support_plk", default=None,
        help="可选：显式指定 support 特征文件"
    )
    parser.add_argument(
        "--query_plk", default=None,
        help="可选：显式指定 query 特征文件"
    )
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


# ============================================================
# GPU
# ============================================================

use_gpu = torch.cuda.is_available()


# ============================================================
# 这些变量后面 PT-MAP 类中会用到
# ============================================================

n_shot = N_SHOT
n_ways = N_WAYS
n_queries = N_QUERIES
n_runs = N_RUNS

n_lsamples = n_ways * n_shot
n_usamples = n_ways * n_queries
n_samples = n_lsamples + n_usamples

ndatas = None
labels = None
n_nfeat = None


# ============================================================
# Data preprocessing
# ============================================================

def centerDatas(datas):

    # support
    datas[:, :n_lsamples] = (
        datas[:, :n_lsamples, :]
        - datas[:, :n_lsamples].mean(
            1,
            keepdim=True
        )
    )

    support_norm = torch.norm(
        datas[:, :n_lsamples, :],
        2,
        2
    )[:, :, None]

    support_norm = torch.clamp(
        support_norm,
        min=1e-12
    )

    datas[:, :n_lsamples] = (
        datas[:, :n_lsamples, :]
        / support_norm
    )

    # query
    datas[:, n_lsamples:] = (
        datas[:, n_lsamples:, :]
        - datas[:, n_lsamples:].mean(
            1,
            keepdim=True
        )
    )

    query_norm = torch.norm(
        datas[:, n_lsamples:, :],
        2,
        2
    )[:, :, None]

    query_norm = torch.clamp(
        query_norm,
        min=1e-12
    )

    datas[:, n_lsamples:] = (
        datas[:, n_lsamples:, :]
        / query_norm
    )

    return datas


def scaleEachUnitaryDatas(datas):

    norms = datas.norm(
        dim=2,
        keepdim=True
    )

    norms = torch.clamp(
        norms,
        min=1e-12
    )

    return datas / norms


def QRreduction(datas):

    # 原代码使用 torch.qr
    # 保留原 PT-MAP 行为
    ndatas_qr = torch.qr(
        datas.permute(0, 2, 1)
    ).R

    ndatas_qr = ndatas_qr.permute(
        0,
        2,
        1
    )

    return ndatas_qr


# ============================================================
# Base model
# ============================================================

class Model:

    def __init__(self, n_ways):
        self.n_ways = n_ways


# ============================================================
# Gaussian Model
# ============================================================

class GaussianModel(Model):

    def __init__(self, n_ways, lam):

        super(
            GaussianModel,
            self
        ).__init__(n_ways)

        self.mus = None

        self.lam = lam


    def clone(self):

        other = GaussianModel(
            self.n_ways,
            self.lam
        )

        other.mus = self.mus.clone()

        return other


    def cuda(self):

        self.mus = self.mus.cuda()


    def initFromLabelledDatas(self):

        global ndatas

        self.mus = (
            ndatas
            .reshape(
                n_runs,
                n_shot + n_queries,
                n_ways,
                n_nfeat
            )[:, :n_shot]
            .mean(1)
        )


    def updateFromEstimate(
        self,
        estimate,
        alpha
    ):

        Dmus = estimate - self.mus

        self.mus = (
            self.mus
            + alpha * Dmus
        )


    def compute_optimal_transport(
        self,
        M,
        r,
        c,
        epsilon=1e-6
    ):

        device = M.device

        r = r.to(device)
        c = c.to(device)

        n_runs_local, n, m = M.shape

        P = torch.exp(
            -self.lam * M
        )

        P /= (
            P
            .reshape(
                n_runs_local,
                -1
            )
            .sum(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        u = torch.zeros(
            n_runs_local,
            n,
            device=device
        )

        maxiters = 1000
        iters = 1

        while (
            torch.max(
                torch.abs(
                    u - P.sum(2)
                )
            ) > epsilon
        ):

            u = P.sum(2)

            P *= (
                r / torch.clamp(
                    u,
                    min=1e-12
                )
            ).view(
                n_runs_local,
                -1,
                1
            )

            P *= (
                c / torch.clamp(
                    P.sum(1),
                    min=1e-12
                )
            ).view(
                n_runs_local,
                1,
                -1
            )

            if iters == maxiters:
                break

            iters += 1

        return P, torch.sum(P * M)


    def getProbas(self):

        global ndatas
        global labels

        # squared distance to centroids
        dist = (
            ndatas.unsqueeze(2)
            - self.mus.unsqueeze(1)
        ).norm(
            dim=3
        ).pow(2)

        p_xj = torch.zeros_like(
            dist
        )

        r = torch.ones(
            n_runs,
            n_usamples
        )

        c = (
            torch.ones(
                n_runs,
                n_ways
            )
            * n_queries
        )

        p_xj_test, _ = (
            self.compute_optimal_transport(
                dist[:, n_lsamples:],
                r,
                c,
                epsilon=1e-6
            )
        )

        p_xj[:, n_lsamples:] = (
            p_xj_test
        )

        # support labels are known
        p_xj[:, :n_lsamples].fill_(0)

        p_xj[:, :n_lsamples].scatter_(
            2,
            labels[
                :, :n_lsamples
            ].unsqueeze(2),
            1
        )

        return p_xj


    def estimateFromMask(
        self,
        mask
    ):

        global ndatas

        denominator = (
            mask
            .sum(dim=1)
            .unsqueeze(2)
        )

        denominator = torch.clamp(
            denominator,
            min=1e-12
        )

        emus = (
            mask
            .permute(0, 2, 1)
            .matmul(ndatas)
            .div(denominator)
        )

        return emus


# ============================================================
# MAP
# ============================================================

class MAP:

    def __init__(
        self,
        alpha=None
    ):

        self.verbose = False

        self.progressBar = False

        self.alpha = alpha


    def getAccuracy(
        self,
        probas
    ):

        olabels = probas.argmax(
            dim=2
        )

        matches = labels.eq(
            olabels
        ).float()

        acc_test = matches[
            :,
            n_lsamples:
        ].mean(1)

        m = acc_test.mean().item()

        pm = (
            acc_test.std().item()
            * 1.96
            / math.sqrt(n_runs)
        )

        return m, pm


    def performEpoch(
        self,
        model,
        epochInfo=None
    ):

        p_xj = model.getProbas()

        self.probas = p_xj

        if self.verbose:

            print(
                "accuracy from filtered probas",
                self.getAccuracy(
                    self.probas
                )
            )

        m_estimates = (
            model.estimateFromMask(
                self.probas
            )
        )

        model.updateFromEstimate(
            m_estimates,
            self.alpha
        )

        if self.verbose:

            op_xj = model.getProbas()

            acc = self.getAccuracy(
                op_xj
            )

            print(
                "output model accuracy",
                acc
            )


    def loop(
        self,
        model,
        n_epochs=20
    ):

        self.probas = (
            model.getProbas()
        )

        if self.verbose:

            print(
                "initialisation model accuracy",
                self.getAccuracy(
                    self.probas
                )
            )

        pb = None

        if self.progressBar:

            if type(
                self.progressBar
            ) == bool:

                pb = tqdm(
                    total=n_epochs
                )

            else:

                pb = self.progressBar

        for epoch in range(
            1,
            n_epochs + 1
        ):

            if self.verbose:

                print(
                    "----- epoch[{:3d}] "
                    "lr_m: {:0.3f}".format(
                        epoch,
                        self.alpha
                    )
                )

            self.performEpoch(
                model,
                epochInfo=(
                    epoch,
                    n_epochs
                )
            )

            if pb is not None:
                pb.update()

        if pb is not None:
            try:
                pb.close()
            except Exception:
                pass

        op_xj = model.getProbas()

        acc = self.getAccuracy(
            op_xj
        )

        return acc


# ============================================================
# Medical feature loading and episodic sampling
# ============================================================

def load_feature_dict(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature file not found:\n  {path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)

    features = {}
    for class_id, values in raw.items():
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(
                f"Invalid features for class {class_id} in {path}: "
                f"shape={array.shape}"
            )
        features[int(class_id)] = array
    return features


def generate_medical_episodes(
    support_features,
    query_features,
    eval_mode,
    ways,
    shot,
    queries,
    runs,
    seed,
):
    """Generate [run, way, shot+query, feature] PT-MAP episodes.

    test-test: support/query both come from the test feature pool. Query
    indices are selected first and excluded from the support candidates.
    Rare-class support sampling uses replacement only when necessary.

    test-train: support comes from train/base and query from test/novel.
    Classes are sampled from the intersection of the two label spaces.
    """
    rng = np.random.default_rng(seed)
    common_classes = sorted(
        set(support_features) & set(query_features)
    )
    if len(common_classes) < ways:
        raise ValueError(
            f"Only {len(common_classes)} common classes are available, "
            f"but n_way={ways}. Common classes: {common_classes}"
        )

    feature_dims = {
        value.shape[1]
        for pool in (support_features, query_features)
        for value in pool.values()
    }
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions: {feature_dims}")

    episodes = []
    chosen_class_ids = []

    for _ in range(runs):
        class_ids = rng.choice(
            np.asarray(common_classes), size=ways, replace=False
        ).tolist()
        episode = []

        for class_id in class_ids:
            support_pool = support_features[int(class_id)]
            query_pool = query_features[int(class_id)]

            if eval_mode == "test-test":
                if support_pool is not query_pool:
                    raise ValueError(
                        "test-test requires the same feature dictionary "
                        "for support and query"
                    )
                if len(query_pool) <= queries:
                    raise ValueError(
                        f"Class {class_id} needs at least {queries + 1} "
                        "distinct test images for disjoint support/query, "
                        f"but has {len(query_pool)}"
                    )

                query_idx = rng.choice(
                    len(query_pool), size=queries, replace=False
                )
                support_candidates = np.setdiff1d(
                    np.arange(len(support_pool)), query_idx,
                    assume_unique=False
                )
                support_idx = rng.choice(
                    support_candidates,
                    size=shot,
                    replace=len(support_candidates) < shot,
                )
            elif eval_mode == "test-train":
                support_idx = rng.choice(
                    len(support_pool), size=shot,
                    replace=len(support_pool) < shot,
                )
                query_idx = rng.choice(
                    len(query_pool), size=queries,
                    replace=len(query_pool) < queries,
                )
            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            class_samples = np.concatenate(
                [support_pool[support_idx], query_pool[query_idx]], axis=0
            )
            episode.append(class_samples)

        episodes.append(np.stack(episode, axis=0))
        chosen_class_ids.append(class_ids)

    return (
        torch.from_numpy(np.stack(episodes, axis=0)).float(),
        chosen_class_ids,
    )


def default_feature_paths(args):
    feature_dir = os.path.join(
        args.feature_root,
        args.medical_subset,
        f"{args.model}_{args.method}",
        "last",
    )

    if args.eval_mode == "test-test":
        query_path = args.query_plk or os.path.join(
            feature_dir, "output.plk"
        )
        support_path = args.support_plk or query_path
    else:
        support_path = args.support_plk or os.path.join(
            feature_dir, "base_output.plk"
        )
        query_path = args.query_plk or os.path.join(
            feature_dir, "novel_output.plk"
        )
    return support_path, query_path


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    cli = parse_args()

    DATASET = "medical"
    EVAL_MODE = cli.eval_mode
    MEDICAL_SUBSET = cli.medical_subset
    MODEL_NAME = cli.model
    METHOD = cli.method
    N_WAYS = cli.n_way
    N_SHOT = cli.n_shot
    N_QUERIES = cli.n_query
    N_RUNS = cli.episodes
    LAM = cli.lam
    ALPHA = cli.alpha
    N_ITER = cli.iterations
    BETA = cli.beta

    n_shot = N_SHOT
    n_ways = N_WAYS
    n_queries = N_QUERIES
    n_runs = N_RUNS
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    if cli.output_json:
        OUTPUT_JSON = cli.output_json
    else:
        OUTPUT_JSON = os.path.join(
            "./results_tsy",
            EVAL_MODE,
            f"{MEDICAL_SUBSET}_{MODEL_NAME}_{METHOD}_ptmap_"
            f"{N_WAYS}way_{N_SHOT}shot_{N_QUERIES}query_"
            f"{N_RUNS}ep.json",
        )

    print()
    print("=" * 80)
    print("PT-MAP TEST")
    print("=" * 80)

    print(
        f"Dataset          : {DATASET}"
    )

    print(
        f"Medical subset   : {MEDICAL_SUBSET}"
    )

    print(
        f"Evaluation mode  : {EVAL_MODE}"
    )

    print(
        f"Backbone         : {MODEL_NAME}"
    )

    print(
        f"Method           : {METHOD}"
    )

    print(
        f"N-way            : {N_WAYS}"
    )

    print(
        f"N-shot           : {N_SHOT}"
    )

    print(
        f"N-query/class    : {N_QUERIES}"
    )

    print(
        f"Episodes         : {N_RUNS}"
    )

    print(
        f"lambda           : {LAM}"
    )

    print(
        f"alpha            : {ALPHA}"
    )

    print(
        f"iterations       : {N_ITER}"
    )

    print(
        f"beta             : {BETA}"
    )

    print(
        f"Output JSON      : {OUTPUT_JSON}"
    )

    print("=" * 80)


    support_path, query_path = default_feature_paths(cli)
    print(f"Support features : {support_path}")
    print(f"Query features   : {query_path}")

    if EVAL_MODE == "test-test":
        query_features = load_feature_dict(query_path)
        support_features = query_features
    else:
        support_features = load_feature_dict(support_path)
        query_features = load_feature_dict(query_path)

    ndatas, episode_class_ids = generate_medical_episodes(
        support_features=support_features,
        query_features=query_features,
        eval_mode=EVAL_MODE,
        ways=n_ways,
        shot=n_shot,
        queries=n_queries,
        runs=n_runs,
        seed=cli.seed,
    )

    print(
        "\nRaw generated episode tensor:",
        ndatas.shape
    )


    # ========================================================
    # Original tensor:
    #
    # [run, way, shot+query, feature]
    #
    # convert:
    #
    # [run, shot+query, way, feature]
    #
    # then:
    #
    # [run, sample, feature]
    # ========================================================

    ndatas = (
        ndatas
        .permute(
            0,
            2,
            1,
            3
        )
        .reshape(
            n_runs,
            n_samples,
            -1
        )
    )


    # ========================================================
    # Labels
    #
    # Episode ordering:
    #
    # support:
    # t0: class0 class1 ... classN
    # t1: class0 class1 ... classN
    # ...
    #
    # query:
    # class0 class1 ... classN
    # ========================================================

    labels = (
        torch
        .arange(n_ways)
        .view(
            1,
            1,
            n_ways
        )
        .expand(
            n_runs,
            n_shot + n_queries,
            n_ways
        )
        .clone()
        .view(
            n_runs,
            n_samples
        )
    )


    # ========================================================
    # Power transform
    # ========================================================

    beta = BETA

    ndatas = torch.pow(
        ndatas + 1e-6,
        beta
    )


    # ========================================================
    # QR reduction
    # ========================================================

    ndatas = QRreduction(
        ndatas
    )

    n_nfeat = ndatas.size(2)


    # ========================================================
    # Normalization
    # ========================================================

    ndatas = (
        scaleEachUnitaryDatas(
            ndatas
        )
    )

    ndatas = centerDatas(
        ndatas
    )


    print(
        "\nProcessed data size:",
        ndatas.size()
    )

    print(
        f"Feature dimension after QR: "
        f"{n_nfeat}"
    )


    # ========================================================
    # GPU
    # ========================================================

    if use_gpu:

        print(
            "\nUsing CUDA"
        )

        ndatas = ndatas.cuda()

        labels = labels.cuda()

    else:

        print(
            "\nUsing CPU"
        )


    # ========================================================
    # PT-MAP
    # ========================================================

    lam = LAM

    ptmap_model = GaussianModel(
        n_ways,
        lam
    )

    ptmap_model.initFromLabelledDatas()


    alpha = ALPHA

    ptmap_optim = MAP(
        alpha
    )

    ptmap_optim.verbose = False

    ptmap_optim.progressBar = True


    print(
        "\nStarting PT-MAP..."
    )


    acc_test = (
        ptmap_optim.loop(
            ptmap_model,
            n_epochs=N_ITER
        )
    )


    # ========================================================
    # Result
    # ========================================================

    accuracy = (
        acc_test[0]
        * 100
    )

    pm = (
        acc_test[1]
        * 100
    )


    print()
    print("=" * 80)

    print(
        "FINAL RESULT"
    )

    print("=" * 80)

    print(
        "Accuracy: "
        "{:0.2f} +- {:0.2f}".format(
            accuracy,
            pm
        )
    )

    print("=" * 80)


    # ========================================================
    # Save JSON
    # ========================================================

    output_dir = os.path.dirname(
        OUTPUT_JSON
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    result = {

        "dataset":
            DATASET,

        "medical_subset":
            MEDICAL_SUBSET,

        "eval_mode":
            EVAL_MODE,

        "support_feature_path":
            os.path.abspath(support_path),

        "query_feature_path":
            os.path.abspath(query_path),

        "seed":
            int(cli.seed),

        "model":
            MODEL_NAME,

        "method":
            METHOD,

        "n_ways":
            n_ways,

        "n_shot":
            n_shot,

        "n_queries":
            n_queries,

        "n_runs":
            n_runs,

        "test_acc":
            float(accuracy),

        "pm":
            float(pm),

        "lam":
            float(LAM),

        "alpha":
            float(ALPHA),

        "beta":
            float(BETA),

        "n_iter":
            int(N_ITER)
    }


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\nSaved result to:"
    )

    print(
        f"  {OUTPUT_JSON}"
    )

    print()
