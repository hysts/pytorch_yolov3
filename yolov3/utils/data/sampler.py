import torch.utils.data.sampler


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self,
                 sampler,
                 batch_size,
                 max_iterations,
                 drop_last=False,
                 start_iter=0):
        """Iteration-based batch sampler

        Args:
            sampler (torch.utils.data.sampler.Sampler): sampler
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            drop_last (bool): whether to drop the last batch whose
                size is smaller than `batch_size`
            start_iter (int): starting iteration
        """
        super().__init__(sampler, batch_size, drop_last)
        self.max_iterations = max_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.max_iterations:
            # in distributed case, a sampler has `set_epoch` method and
            # uses the passed number as a random seed to shuffle dataset
            # indices
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(iteration)
            for batch in super().__iter__():
                iteration += 1
                if iteration > self.max_iterations:
                    break
                yield batch

    def __len__(self):
        return self.max_iterations
