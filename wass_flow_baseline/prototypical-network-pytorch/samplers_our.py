import torch
import numpy as np


class CategoriesSampler:

    def __init__(self, label, n_batch, n_cls, n_per):
        """
        Prototypical Network episodic sampler.

        Parameters
        ----------
        label : list / np.ndarray
            Label of every image in the dataset.

        n_batch : int
            Number of episodes.

        n_cls : int
            Number of classes per episode (N-way).

        n_per : int
            Number of samples per class in each episode.

            For example:

                shot = 5
                query = 1

            then:

                n_per = 5 + 1 = 6


        Sampling protocol
        -----------------
        This implementation assumes:

            n_query = 1
            n_support = n_per - 1

        For each selected class:

        1. Randomly select one QUERY sample.
        2. Remove that query sample from the SUPPORT candidate pool.
        3. Sample support images:
              - enough candidates:
                    without replacement
              - insufficient candidates:
                    with replacement
        4. Store samples in this order:

              [support_1,
               support_2,
               ...,
               support_K,
               query]

        After:

            torch.stack(batch).t().reshape(-1)

        the final episode ordering is:

            support position 1 for all classes
            support position 2 for all classes
            ...
            support position K for all classes
            query for all classes

        This preserves the standard ProtoNet episode layout.
        """

        self.n_batch = int(n_batch)
        self.n_cls = int(n_cls)
        self.n_per = int(n_per)

        # ============================================================
        # Current protocol:
        #
        # n_per = n_support + 1 query
        #
        # Example:
        #   5-shot + 1-query
        #   n_per = 6
        # ============================================================
        self.n_query = 1
        self.n_support = self.n_per - self.n_query

        if self.n_per < 2:
            raise ValueError(
                f"n_per must be >= 2, but got n_per={self.n_per}"
            )

        if self.n_cls < 1:
            raise ValueError(
                f"n_cls must be >= 1, but got n_cls={self.n_cls}"
            )

        # ============================================================
        # Convert labels to numpy
        # ============================================================
        label = np.asarray(label)

        if label.size == 0:
            raise ValueError("Empty label array.")

        # ============================================================
        # Build class -> sample-index mapping
        # ============================================================
        self.m_ind = []

        max_label = int(label.max())

        for i in range(max_label + 1):

            ind = np.argwhere(
                label == i
            ).reshape(-1)

            ind = torch.from_numpy(
                ind
            ).long()

            self.m_ind.append(
                ind
            )

        # ============================================================
        # Valid classes
        #
        # Since query and support must not use the same original image:
        #
        # minimum requirement:
        #
        #   1 query
        #   1 distinct support candidate
        #
        # therefore:
        #
        #   class size >= 2
        # ============================================================
        self.valid_classes = [
            class_id
            for class_id, indices in enumerate(self.m_ind)
            if len(indices) >= 2
        ]

        if len(self.valid_classes) < self.n_cls:
            raise ValueError(
                "\nNot enough valid classes for episodic sampling.\n"
                f"Requested n_cls = {self.n_cls}\n"
                f"Valid classes   = {len(self.valid_classes)}\n"
                f"Requirement     = each class must contain >= 2 images\n"
                f"Valid class ids = {self.valid_classes}\n"
            )

        # ============================================================
        # Debug information
        # ============================================================
        print("=" * 80)
        print("[CategoriesSampler]")
        print(f"n_batch        = {self.n_batch}")
        print(f"n_cls          = {self.n_cls}")
        print(f"n_per          = {self.n_per}")
        print(f"n_support      = {self.n_support}")
        print(f"n_query        = {self.n_query}")
        print(f"total classes  = {len(self.m_ind)}")
        print(f"valid classes  = {len(self.valid_classes)}")

        rare_classes = []

        for class_id in self.valid_classes:

            n_available = len(
                self.m_ind[class_id]
            )

            # For 5-shot + 1-query, ideal class size >= 6.
            #
            # If class size < 6 but >=2:
            # support sampling will use replacement.
            if n_available < self.n_per:

                rare_classes.append(
                    (
                        class_id,
                        n_available
                    )
                )

        if rare_classes:

            print(
                "[Rare Classes] "
                "support replacement will be used:"
            )

            for class_id, n_available in rare_classes:

                print(
                    f"  class={class_id}, "
                    f"available={n_available}, "
                    f"required={self.n_per}"
                )

        else:

            print(
                "[Rare Classes] None. "
                "All classes have enough images."
            )

        print("=" * 80)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):

            batch = []

            # ========================================================
            # 1. Randomly select N classes from valid classes
            # ========================================================
            class_order = torch.randperm(
                len(self.valid_classes)
            )[:self.n_cls]

            classes = [
                self.valid_classes[
                    idx
                ]
                for idx in class_order.tolist()
            ]

            # ========================================================
            # 2. Sample one episode for every selected class
            # ========================================================
            for c in classes:

                # All image indices belonging to current class
                l = self.m_ind[c]

                n_available = len(l)

                # This should already be prevented by valid_classes.
                if n_available < 2:

                    raise RuntimeError(
                        f"Class {c} only contains "
                        f"{n_available} image(s). "
                        f"Cannot keep query/support disjoint."
                    )

                # ====================================================
                # Step A:
                # Sample ONE query first
                # ====================================================
                query_pos = torch.randint(
                    low=0,
                    high=n_available,
                    size=(1,),
                )

                query_idx = l[
                    query_pos
                ]

                # ====================================================
                # Step B:
                # Exclude query from support candidate pool
                # ====================================================
                support_mask = torch.ones(
                    n_available,
                    dtype=torch.bool,
                )

                support_mask[
                    query_pos.item()
                ] = False

                support_pool = l[
                    support_mask
                ]

                if len(support_pool) == 0:

                    raise RuntimeError(
                        f"Class {c} has no support candidates "
                        f"after excluding query."
                    )

                # ====================================================
                # Step C:
                # Sample SUPPORT
                #
                # Enough candidates:
                #     without replacement
                #
                # Not enough candidates:
                #     with replacement
                # ====================================================
                if len(support_pool) >= self.n_support:

                    support_pos = torch.randperm(
                        len(support_pool)
                    )[:self.n_support]

                    support_idx = support_pool[
                        support_pos
                    ]

                else:

                    support_pos = torch.randint(
                        low=0,
                        high=len(support_pool),
                        size=(self.n_support,),
                    )

                    support_idx = support_pool[
                        support_pos
                    ]

                # ====================================================
                # Step D:
                # Keep:
                #
                # support first
                # query last
                #
                # Example:
                #
                # 5-shot + 1-query:
                #
                # [S1, S2, S3, S4, S5, Q]
                # ====================================================
                selected = torch.cat(
                    [
                        support_idx,
                        query_idx,
                    ],
                    dim=0,
                )

                # Safety check
                if len(selected) != self.n_per:

                    raise RuntimeError(
                        "\nInternal sampler size mismatch.\n"
                        f"class={c}\n"
                        f"selected={len(selected)}\n"
                        f"expected={self.n_per}\n"
                    )

                # ====================================================
                # Extra safety:
                # Query must NOT appear in support
                # ====================================================
                query_value = int(
                    query_idx.item()
                )

                if any(
                    int(x.item()) == query_value
                    for x in support_idx
                ):

                    raise RuntimeError(
                        "\nQuery leakage detected.\n"
                        f"class={c}\n"
                        f"query_idx={query_value}\n"
                        f"support_idx={support_idx.tolist()}\n"
                    )

                batch.append(
                    selected
                )

            # ========================================================
            # Original ProtoNet episode ordering
            # ========================================================
            #
            # Before transpose:
            #
            # class 0:
            #   S1 S2 S3 S4 S5 Q
            #
            # class 1:
            #   S1 S2 S3 S4 S5 Q
            #
            # ...
            #
            # shape:
            #
            #   [n_cls, n_per]
            #
            #
            # After transpose:
            #
            #   S1(class0)
            #   S1(class1)
            #   ...
            #
            #   S2(class0)
            #   S2(class1)
            #   ...
            #
            #   ...
            #
            #   Q(class0)
            #   Q(class1)
            #   ...
            #
            # Therefore:
            #
            # first:
            #   n_support * n_cls
            #
            # are support samples
            #
            # final:
            #   n_query * n_cls
            #
            # are query samples
            # ========================================================
            batch = torch.stack(
                batch,
                dim=0
            )

            # shape:
            # [n_cls, n_per]
            if batch.shape != (
                self.n_cls,
                self.n_per
            ):

                raise RuntimeError(
                    "\nUnexpected episode tensor shape.\n"
                    f"got={tuple(batch.shape)}\n"
                    f"expected=({self.n_cls}, {self.n_per})\n"
                )

            batch = batch.t().reshape(
                -1
            )

            yield batch