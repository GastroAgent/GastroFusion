
import numpy as np
import torch


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)

        self.m_ind = []

        for i in range(max(label) + 1):

            ind = np.argwhere(label == i).reshape(-1)

            self.m_ind.append(
                torch.from_numpy(ind)
            )


    def __len__(self):
        return self.n_batch


    def __iter__(self):

        for _ in range(self.n_batch):

            batch = []

            # sample classes
            classes = torch.randperm(
                len(self.m_ind)
            )[:self.n_cls]


            for c in classes:

                l = self.m_ind[c]

                num = len(l)


                # ===============================
                # key modification:
                # allow replacement when rare class
                # ===============================

                if num >= self.n_per:

                    pos = torch.randperm(num)[:self.n_per]

                else:

                    pos = torch.randint(
                        low=0,
                        high=num,
                        size=(self.n_per,)
                    )


                batch.append(
                    l[pos]
                )


            yield torch.stack(batch).t().reshape(-1)



class CrossDomainCategoriesSampler:
    """
    support/query from different splits.

    Huaxi test-train:

        support: train
        query:   test

    """

    def __init__(
        self,
        support_label,
        query_label,
        n_batch,
        n_cls,
        k_shot,
        k_query
    ):

        self.n_batch = n_batch
        self.n_cls = n_cls

        self.k_shot = k_shot
        self.k_query = k_query


        self.support_label = np.array(
            support_label
        )

        self.query_label = np.array(
            query_label
        )


        self.classes = sorted(
            list(
                set(self.support_label)
                &
                set(self.query_label)
            )
        )


        self.support_index = {

            c: np.where(
                self.support_label == c
            )[0]

            for c in self.classes

        }


        self.query_index = {

            c: np.where(
                self.query_label == c
            )[0]

            for c in self.classes

        }



    def __len__(self):

        return self.n_batch



    def __iter__(self):

        for _ in range(self.n_batch):


            classes = np.random.choice(
                self.classes,
                self.n_cls,
                replace=False
            )


            support_idx = []
            query_idx = []


            for c in classes:


                support_pool = self.support_index[c]

                query_pool = self.query_index[c]


                # ===============================
                # support sampling
                # ===============================

                s = np.random.choice(
                    support_pool,
                    self.k_shot,
                    replace=(
                        len(support_pool)
                        <
                        self.k_shot
                    )
                )


                # ===============================
                # query sampling
                # ===============================

                q = np.random.choice(
                    query_pool,
                    self.k_query,
                    replace=(
                        len(query_pool)
                        <
                        self.k_query
                    )
                )


                support_idx.extend(
                    s.tolist()
                )

                query_idx.extend(
                    q.tolist()
                )


            yield (

                torch.tensor(
                    support_idx
                ),

                torch.tensor(
                    query_idx
                )

            )