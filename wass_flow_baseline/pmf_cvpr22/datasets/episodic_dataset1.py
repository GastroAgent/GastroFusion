import os
import json
import numpy as np
import torch
import torch.utils.data as data
import PIL.Image as Image

from torchvision.datasets import ImageFolder


# ============================================================
# Basic image loader
# ============================================================

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def PilLoaderRGB(imgPath):
    return Image.open(imgPath).convert("RGB")


def _is_image_file(path):
    return (
        os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )


def _list_images(cls_dir):
    """
    Return image filenames under one class folder.
    """
    images = []

    for name in sorted(os.listdir(cls_dir)):
        path = os.path.join(cls_dir, name)

        if _is_image_file(path):
            images.append(name)

    return images


def _path_key(path):
    """
    Pure string normalization.

    Used to prevent query/support from accidentally referring
    to the exact same file.
    """
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(
                os.path.expanduser(path)
            )
        )
    )


# ============================================================
# Episodic Dataset
# ============================================================

class EpisodeDataset(data.Dataset):
    """
    Episodic Few-Shot Dataset with separate Query / Support pools.

    ============================================================
    Test-Train protocol
    ============================================================

    Query:
        queryDir/<class>/*

    Support:
        supportDir/<class>/*

    Candidate classes:
        query classes ∩ support classes

    A class is retained as long as:

        query side   >= 1 image
        support side >= 1 image

    ============================================================
    K-shot shortage policy
    ============================================================

    If support candidates >= nSupport:
        sample WITHOUT replacement

    If support candidates < nSupport:
        sample WITH replacement

    Example:

        support pool = [A, B, C]
        nSupport = 5

        possible support:
            A B C A B

    Query and support are independent because they normally come
    from different directories.

    Nevertheless, if the same physical path appears on both sides,
    the query path is explicitly excluded from support.

    ============================================================
    Query shortage policy
    ============================================================

    Canonical evaluation uses nQuery=1.

    Therefore a class with only ONE query image is fully valid.

    If nQuery > available query images, query sampling falls back
    to sampling with replacement.
    """

    def __init__(
        self,
        queryDir,
        supportDir,
        nCls,
        nSupport,
        nQuery,
        transform,
        inputW,
        inputH,
        nEpisode=2000,
    ):
        super().__init__()

        self.queryDir = queryDir
        self.supportDir = supportDir

        self.nSupport = int(nSupport)
        self.nQuery = int(nQuery)
        self.transform = transform
        self.nEpisode = int(nEpisode)

        self.inputW = int(inputW)
        self.inputH = int(inputH)

        # ========================================================
        # Check roots
        # ========================================================

        if not os.path.isdir(self.queryDir):
            raise FileNotFoundError(
                f"Query directory not found: {self.queryDir}"
            )

        if not os.path.isdir(self.supportDir):
            raise FileNotFoundError(
                f"Support directory not found: {self.supportDir}"
            )

        # ========================================================
        # Scan class folders
        # ========================================================

        queryClasses = {
            name
            for name in os.listdir(self.queryDir)
            if os.path.isdir(
                os.path.join(self.queryDir, name)
            )
        }

        supportClasses = {
            name
            for name in os.listdir(self.supportDir)
            if os.path.isdir(
                os.path.join(self.supportDir, name)
            )
        }

        # ========================================================
        # Candidate classes = query ∩ support
        #
        # Keep class if:
        #   query   >= 1 image
        #   support >= 1 image
        # ========================================================

        self.queryImageLists = {}
        self.supportImageLists = {}

        validClasses = []

        for cls in sorted(queryClasses & supportClasses):

            queryClsPath = os.path.join(
                self.queryDir,
                cls,
            )

            supportClsPath = os.path.join(
                self.supportDir,
                cls,
            )

            queryImgs = _list_images(
                queryClsPath
            )

            supportImgs = _list_images(
                supportClsPath
            )

            if len(queryImgs) < 1:
                continue

            if len(supportImgs) < 1:
                continue

            validClasses.append(cls)

            self.queryImageLists[cls] = queryImgs
            self.supportImageLists[cls] = supportImgs

        self.clsList = validClasses

        # ========================================================
        # Effective N-way
        #
        # For reduced Test-Train:
        # use all shared classes, up to requested nCls.
        #
        # Example:
        # requested = 8
        # shared     = 6
        # effective  = 6
        # ========================================================

        requested_nCls = int(nCls)

        self.nCls = min(
            requested_nCls,
            len(self.clsList)
        )

        if self.nCls < 2:
            raise ValueError(
                "\nNot enough shared Query/Support classes.\n"
                f"queryDir   = {self.queryDir}\n"
                f"supportDir = {self.supportDir}\n"
                f"shared valid classes = {len(self.clsList)}\n"
                f"classes = {self.clsList}\n"
            )

        if self.nCls < requested_nCls:
            print(
                f"[EpisodeDataset] "
                f"requested={requested_nCls}-way, "
                f"but only {len(self.clsList)} valid shared classes exist. "
                f"Use effective {self.nCls}-way.",
                flush=True,
            )

        # If there are more valid classes than requested,
        # each episode samples nCls of them.
        if len(self.clsList) > self.nCls:
            print(
                f"[EpisodeDataset] "
                f"valid shared classes={len(self.clsList)}, "
                f"episode way={self.nCls}.",
                flush=True,
            )

        # ========================================================
        # Print class statistics once
        # ========================================================

        print("=" * 100)
        print("[EpisodeDataset Test-Train]")
        print(f"Query root        : {self.queryDir}")
        print(f"Support root      : {self.supportDir}")
        print(f"Requested way     : {requested_nCls}")
        print(f"Effective way     : {self.nCls}")
        print(f"Shot              : {self.nSupport}")
        print(f"Query per class   : {self.nQuery}")
        print(f"Episodes          : {self.nEpisode}")
        print("Valid classes:")

        for cls in self.clsList:
            print(
                f"  {cls}: "
                f"query={len(self.queryImageLists[cls])}, "
                f"support={len(self.supportImageLists[cls])}"
            )

        print("=" * 100)

        # ========================================================
        # Tensor buffers
        # ========================================================

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(
            self.nCls * self.nSupport,
            3,
            self.inputW,
            self.inputH,
        )

        self.labelSupport = intType(
            self.nCls * self.nSupport
        )

        self.tensorQuery = floatType(
            self.nCls * self.nQuery,
            3,
            self.inputW,
            self.inputH,
        )

        self.labelQuery = intType(
            self.nCls * self.nQuery
        )

        self.imgTensor = floatType(
            3,
            self.inputW,
            self.inputH,
        )

        # ========================================================
        # Labels {0, ..., nCls-1}
        # ========================================================

        for i in range(self.nCls):

            self.labelSupport[
                i * self.nSupport:
                (i + 1) * self.nSupport
            ] = i

            self.labelQuery[
                i * self.nQuery:
                (i + 1) * self.nQuery
            ] = i

    def __len__(self):
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Generate one episode.

        For every selected class:

            Query:
                sampled from queryDir

            Support:
                sampled from supportDir

        Support shortage:
            sample with replacement.

        Query with only one image:
            perfectly valid for nQuery=1.
        """

        # ========================================================
        # 1. Select episode classes
        # ========================================================

        clsEpisode = np.random.choice(
            self.clsList,
            self.nCls,
            replace=False,
        )

        # ========================================================
        # 2. Build query/support for each class
        # ========================================================

        for i, cls in enumerate(clsEpisode):

            queryClsPath = os.path.join(
                self.queryDir,
                cls,
            )

            supportClsPath = os.path.join(
                self.supportDir,
                cls,
            )

            queryList = self.queryImageLists[cls]
            supportList = self.supportImageLists[cls]

            # ====================================================
            # Query sampling
            # ====================================================

            queryReplace = (
                len(queryList) < self.nQuery
            )

            queryImgs = np.random.choice(
                queryList,
                self.nQuery,
                replace=queryReplace,
            ).tolist()

            # ====================================================
            # Build support candidates
            #
            # Normally query/support roots are different.
            # Still explicitly remove an identical physical path
            # if one happens to appear.
            # ====================================================

            queryFullPaths = {
                _path_key(
                    os.path.join(
                        queryClsPath,
                        img,
                    )
                )
                for img in queryImgs
            }

            supportCandidates = []

            for img in supportList:

                supportPath = os.path.join(
                    supportClsPath,
                    img,
                )

                if _path_key(supportPath) in queryFullPaths:
                    continue

                supportCandidates.append(img)

            if len(supportCandidates) < 1:
                raise RuntimeError(
                    f"\nEpisode {idx}: "
                    f"class '{cls}' has no usable support image "
                    f"after excluding query.\n"
                    f"query={queryImgs}\n"
                    f"queryDir={queryClsPath}\n"
                    f"supportDir={supportClsPath}\n"
                )

            # ====================================================
            # Support sampling
            #
            # enough:
            #     WITHOUT replacement
            #
            # shortage:
            #     WITH replacement
            # ====================================================

            supportReplace = (
                len(supportCandidates) < self.nSupport
            )

            supportImgs = np.random.choice(
                supportCandidates,
                self.nSupport,
                replace=supportReplace,
            ).tolist()

            # ====================================================
            # Load support
            # ====================================================

            for j, img in enumerate(supportImgs):

                imgPath = os.path.join(
                    supportClsPath,
                    img,
                )

                I = PilLoaderRGB(
                    imgPath
                )

                transformed = self.transform(I)

                self.tensorSupport[
                    i * self.nSupport + j
                ] = self.imgTensor.copy_(
                    transformed
                )

            # ====================================================
            # Load query
            # ====================================================

            for j, img in enumerate(queryImgs):

                imgPath = os.path.join(
                    queryClsPath,
                    img,
                )

                I = PilLoaderRGB(
                    imgPath
                )

                transformed = self.transform(I)

                self.tensorQuery[
                    i * self.nQuery + j
                ] = self.imgTensor.copy_(
                    transformed
                )

        # ========================================================
        # Randomly shuffle support/query examples
        #
        # Labels are shuffled with exactly the same permutation.
        # ========================================================

        permSupport = torch.randperm(
            self.nCls * self.nSupport
        )

        permQuery = torch.randperm(
            self.nCls * self.nQuery
        )

        return (
            self.tensorSupport[permSupport],
            self.labelSupport[permSupport],
            self.tensorQuery[permQuery],
            self.labelQuery[permQuery],
        )


# ============================================================
# JSON episode dataset
# ============================================================

class EpisodeJSONDataset(data.Dataset):
    """
    Fixed episodes from JSON.
    """

    def __init__(
        self,
        episodeJson,
        imgDir,
        inputW,
        inputH,
        valTransform,
    ):
        with open(
            episodeJson,
            "r",
            encoding="utf-8",
        ) as f:
            self.episodeInfo = json.load(f)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)

        self.nCls = len(
            self.episodeInfo[0]["Support"]
        )

        self.nSupport = len(
            self.episodeInfo[0]["Support"][0]
        )

        self.nQuery = len(
            self.episodeInfo[0]["Query"][0]
        )

        self.transform = valTransform

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(
            self.nCls * self.nSupport,
            3,
            inputW,
            inputH,
        )

        self.labelSupport = intType(
            self.nCls * self.nSupport
        )

        self.tensorQuery = floatType(
            self.nCls * self.nQuery,
            3,
            inputW,
            inputH,
        )

        self.labelQuery = intType(
            self.nCls * self.nQuery
        )

        self.imgTensor = floatType(
            3,
            inputW,
            inputH,
        )

        for i in range(self.nCls):

            self.labelSupport[
                i * self.nSupport:
                (i + 1) * self.nSupport
            ] = i

            self.labelQuery[
                i * self.nQuery:
                (i + 1) * self.nQuery
            ] = i

    def __getitem__(self, index):

        for i in range(self.nCls):

            for j in range(self.nSupport):

                imgPath = os.path.join(
                    self.imgDir,
                    self.episodeInfo[index]["Support"][i][j],
                )

                I = PilLoaderRGB(imgPath)

                self.tensorSupport[
                    i * self.nSupport + j
                ] = self.imgTensor.copy_(
                    self.transform(I)
                )

            for j in range(self.nQuery):

                imgPath = os.path.join(
                    self.imgDir,
                    self.episodeInfo[index]["Query"][i][j],
                )

                I = PilLoaderRGB(imgPath)

                self.tensorQuery[
                    i * self.nQuery + j
                ] = self.imgTensor.copy_(
                    self.transform(I)
                )

        return (
            self.tensorSupport,
            self.labelSupport,
            self.tensorQuery,
            self.labelQuery,
        )

    def __len__(self):
        return self.nEpisode


# ============================================================
# Standard loader
# ============================================================

def TrainLoader(
    batchSize,
    imgDir,
    trainTransform,
):
    dataloader = data.DataLoader(
        ImageFolder(
            imgDir,
            trainTransform,
        ),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    return dataloader