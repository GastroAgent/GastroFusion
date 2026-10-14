import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        """
        Episodic sampler for DeepEMD.

        Parameters
        ----------
        label : list / np.ndarray
            Label of every image in the dataset.

        n_batch : int
            Number of episodes.

        n_cls : int
            Number of classes per episode, i.e. N-way.

        n_per : int
            Number of samples per class in one episode.

            In the current DeepEMD setting:

                n_per = n_shot + n_query

            Current experiments use:

                5-shot 1-query

            so:

                n_per = 6

        Sampling strategy
        -----------------
        For each selected class:

        1. Sample query first WITHOUT replacement.
        2. Remove query images from the support candidate pool.
        3. Sample support from the remaining images.
        4. If remaining support candidates are fewer than n_support,
           sample support WITH replacement.
        5. Query and support are therefore guaranteed not to overlap.

        Important
        ---------
        Current code assumes n_query = 1.

        This matches the current experimental setting:
            N-way 5-shot 1-query.
        """

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        # ================================================================
        # Current experimental protocol:
        # 5-shot 1-query
        #
        # n_per = shot + query
        # therefore:
        #   query = 1
        #   support = n_per - 1
        # ================================================================
        self.n_query = 1
        self.n_support = self.n_per - self.n_query

        if self.n_support < 1:
            raise ValueError(
                "n_per must be at least 2 because the current sampler "
                "requires at least 1 support and 1 query image."
            )

        # Convert all labels to numpy.
        label = np.array(label)

        # ================================================================
        # Store dataset indices for every class.
        #
        # self.m_ind[c] contains all dataset indices belonging to class c.
        # ================================================================
        self.m_ind = []

        for i in range(max(label) + 1):

            ind = np.argwhere(label == i).reshape(-1)

            ind = torch.from_numpy(ind)

            self.m_ind.append(ind)

        # ================================================================
        # Valid classes
        #
        # To guarantee support/query disjointness, a class needs at least:
        #
        #   n_query + 1
        #
        # unique images.
        #
        # Example:
        #   n_query = 1
        #
        # A class with 2 images is still usable:
        #
        #   query  = one image
        #   support candidate = the other image
        #
        # and support can then be sampled with replacement.
        # ================================================================
        self.valid_classes = []

        for class_id, class_indices in enumerate(self.m_ind):

            if len(class_indices) >= self.n_query + 1:

                self.valid_classes.append(class_id)

        if len(self.valid_classes) < self.n_cls:

            class_counts = {
                i: len(indices)
                for i, indices in enumerate(self.m_ind)
            }

            raise ValueError(
                "\n"
                "Not enough valid classes for episodic sampling.\n"
                "Requested N-way: {}\n"
                "Valid classes: {}\n"
                "Each valid class must contain at least {} unique images.\n"
                "Class counts: {}\n".format(
                    self.n_cls,
                    len(self.valid_classes),
                    self.n_query + 1,
                    class_counts
                )
            )

    def __len__(self):

        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):

            # ============================================================
            # Randomly select N valid classes for this episode.
            # ============================================================
            class_perm = torch.randperm(
                len(self.valid_classes)
            )[:self.n_cls]

            classes = [
                self.valid_classes[idx.item()]
                for idx in class_perm
            ]

            support_batch = []
            query_batch = []

            # ============================================================
            # Sample support/query for every selected class.
            # ============================================================
            for c in classes:

                class_indices = self.m_ind[c]

                num_images = len(class_indices)

                if num_images < self.n_query + 1:

                    raise RuntimeError(
                        "Class {} unexpectedly contains only {} images. "
                        "At least {} are required.".format(
                            c,
                            num_images,
                            self.n_query + 1
                        )
                    )

                # ========================================================
                # STEP 1
                # Sample QUERY first.
                #
                # Query sampling is always WITHOUT replacement.
                # ========================================================
                query_positions = torch.randperm(
                    num_images
                )[:self.n_query]

                query_indices = class_indices[
                    query_positions
                ]

                # ========================================================
                # STEP 2
                # Exclude query images from support candidates.
                # ========================================================
                support_mask = torch.ones(
                    num_images,
                    dtype=torch.bool
                )

                support_mask[
                    query_positions
                ] = False

                support_candidates = class_indices[
                    support_mask
                ]

                if len(support_candidates) == 0:

                    raise RuntimeError(
                        "Class {} has no support candidates after "
                        "excluding query images.".format(c)
                    )

                # ========================================================
                # STEP 3
                # Sample SUPPORT.
                #
                # Case A:
                # Enough support candidates.
                #
                #     -> sample WITHOUT replacement
                #
                # Case B:
                # Fewer candidates than required support shots.
                #
                #     -> sample WITH replacement
                #
                # Importantly, support replacement happens only inside
                # the candidate pool after removing query images.
                #
                # Therefore support/query cannot overlap.
                # ========================================================
                if len(support_candidates) >= self.n_support:

                    support_positions = torch.randperm(
                        len(support_candidates)
                    )[:self.n_support]

                    support_indices = support_candidates[
                        support_positions
                    ]

                else:

                    support_positions = torch.randint(
                        low=0,
                        high=len(support_candidates),
                        size=(self.n_support,)
                    )

                    support_indices = support_candidates[
                        support_positions
                    ]

                # ========================================================
                # STEP 4
                # Hard safety check:
                #
                # query ∩ support must be empty.
                # ========================================================
                query_set = set(
                    query_indices.tolist()
                )

                support_set = set(
                    support_indices.tolist()
                )

                overlap = query_set & support_set

                if len(overlap) > 0:

                    raise RuntimeError(
                        "Support/query overlap detected.\n"
                        "Episode: {}\n"
                        "Class: {}\n"
                        "Overlap indices: {}\n".format(
                            i_batch,
                            c,
                            sorted(overlap)
                        )
                    )

                support_batch.append(
                    support_indices
                )

                query_batch.append(
                    query_indices
                )

            # ============================================================
            # IMPORTANT:
            #
            # DeepEMD train_meta.py assumes:
            #
            #     k = way * shot
            #
            #     data_shot  = data[:k]
            #     data_query = data[k:]
            #
            # Therefore sampler output MUST be:
            #
            #     all support first
            #     all query second
            #
            # Inside support/query blocks, transpose keeps the original
            # DeepEMD class ordering:
            #
            #     abcdefg abcdefg ...
            #
            # instead of:
            #
            #     aaaaa bbbbb ccccc ...
            # ============================================================

            support_batch = torch.stack(
                support_batch
            ).t().reshape(-1)

            query_batch = torch.stack(
                query_batch
            ).t().reshape(-1)

            batch = torch.cat(
                [
                    support_batch,
                    query_batch
                ],
                dim=0
            )

            # ============================================================
            # Final sanity check.
            #
            # Expected episode size:
            #
            #     n_cls * n_per
            #
            # Example:
            #
            #     7-way 5-shot 1-query
            #
            #     7 * 6 = 42 images
            # ============================================================
            expected_size = (
                self.n_cls *
                self.n_per
            )

            if len(batch) != expected_size:

                raise RuntimeError(
                    "Episode size mismatch: expected {}, got {}.".format(
                        expected_size,
                        len(batch)
                    )
                )

            yield batch