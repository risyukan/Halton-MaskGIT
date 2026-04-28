import torch
import random
import math
import numpy as np
from tqdm import tqdm


class HaltonSampler(object):
    """
    Halton Sampler is a sampling strategy for iterative masked token prediction in image generation models.

    It follows a Halton-based scheduling approach to determine which tokens to predict at each step.
    """

    def __init__(self, sm_temp_min=1, sm_temp_max=1, temp_pow=1, w=4, sched_pow=2.5, step=64, randomize=False, top_k=-1, temp_warmup=0):
        """
        Initializes the HaltonSampler with configurable parameters.

        params:
            sm_temp_min  -> float: Minimum softmax temperature.
            sm_temp_max  -> float: Maximum softmax temperature.
            temp_pow     -> float: Exponent for temperature scheduling.
            w            -> float: Weight parameter for the CFG.
            sched_pow    -> float: Exponent for mask scheduling.
            step         -> int: Number of steps in the sampling process.
            randomize    -> bool: Whether to randomize the Halton sequence for the generation.
            top_k        -> int: If > 0, applies top-k sampling for token selection.
            temp_warmup  -> int: Number of initial steps where temperature is reduced.
        """
        super().__init__()
        self.sm_temp_min = sm_temp_min
        self.sm_temp_max = sm_temp_max
        self.temp_pow = temp_pow
        self.w = w
        self.sched_pow = sched_pow
        self.step = step
        self.randomize = randomize
        self.top_k = top_k
        self.basic_halton_mask = None  # Placeholder for the Halton-based mask
        self.temp_warmup = temp_warmup
        # Linearly interpolate the temperature over the sampling steps
        self.temperature = torch.linspace(self.sm_temp_min, self.sm_temp_max, self.step)

    def __str__(self):
        """Returns a string representation of the sampler configuration."""
        return f"Scheduler: halton, Steps: {self.step}, " \
               f"sm_temp_min: {self.sm_temp_min}, sm_temp_max: {self.sm_temp_max}, w: {self.w}, " \
               f"Top_k: {self.top_k}, temp_warmup: {self.temp_warmup}"

    # ------------------------------------------------------------------
    # Schedule pre-computation
    # ------------------------------------------------------------------

    def compute_schedule(self, input_size, nb_sample=1):
        """Pre-compute U_t and M_t masks for every decoding step.

        U_t  — newly unmasked (released) tokens at step t:  bool (nb_sample, h, w)
        M_t  — all tokens unmasked up to and including step t: bool (nb_sample, h, w)

        This uses the non-randomized Halton order.  Call before __call__ when
        you need the full schedule for e.g. partial-update planning.

        Returns:
            l_U_t: list[Tensor] length step, each (nb_sample, input_size, input_size)
            l_M_t: list[Tensor] length step, each (nb_sample, input_size, input_size)
        """
        if self.basic_halton_mask is None:
            self.basic_halton_mask = self.build_halton_mask(input_size)

        # shape (nb_sample, n_tokens, 2)  — same order for every sample
        halton_mask = (
            self.basic_halton_mask.clone()
            .unsqueeze(0)
            .expand(nb_sample, input_size ** 2, 2)
        )

        l_U_t, l_M_t = [], []
        prev_r = 0
        for index in range(self.step):
            ratio = (index + 1) / self.step
            r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
            r = int(r * (input_size ** 2))
            r = max(index + 1, r)

            # U_t: token coordinates newly released at this step
            _u = halton_mask[:, prev_r:r]   # (nb_sample, n_new, 2)
            U_t = torch.zeros(nb_sample, input_size, input_size, dtype=torch.bool)
            for i in range(nb_sample):
                U_t[i, _u[i, :, 0], _u[i, :, 1]] = True

            # M_t: all released token coordinates up to this step
            _m = halton_mask[:, :r]         # (nb_sample, r, 2)
            M_t = torch.zeros(nb_sample, input_size, input_size, dtype=torch.bool)
            for i in range(nb_sample):
                M_t[i, _m[i, :, 0], _m[i, :, 1]] = True

            l_U_t.append(U_t)
            l_M_t.append(M_t)
            prev_r = r

        return l_U_t, l_M_t

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None,
                 verbose=True, partial_update=False):
        """
        Runs the Halton-based sampling process.

        Args:
            trainer        -> MaskGIT: The model trainer.
            init_code      -> torch.Tensor: Pre-initialized latent code.
            nb_sample      -> int: Number of images to generate.
            labels         -> torch.Tensor: Class labels for conditional generation.
            verbose        -> bool: Whether to display progress.
            partial_update -> bool: If True, pass active_mask (U_t) to the
                             transformer so attention uses Q-only-active mode.

        Returns:
            Tuple: (generated images,
                    list of per-step predicted codes,
                    l_U_t — list of per-step newly-released masks,
                    l_M_t — list of per-step cumulative released masks)
        """

        # Build the Halton mask if not already created
        if self.basic_halton_mask is None:
            self.basic_halton_mask = self.build_halton_mask(trainer.input_size)

        trainer.vit.eval()
        l_codes = []   # intermediate predicted codes
        l_U_t = []     # per-step newly-released token mask  (U_t)
        l_M_t = []     # per-step cumulative released mask   (M_t)

        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(nb_sample - 9)]
                labels = torch.LongTensor(labels[:nb_sample]).to(trainer.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(trainer.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
            else:  # Initialize a code
                code = torch.full((nb_sample, trainer.input_size, trainer.input_size),
                                  trainer.args.mask_value).to(trainer.args.device)

            # Randomizing the mask sequence if enabled
            if self.randomize:
                randomize_mask = torch.randint(0, trainer.input_size ** 2, (nb_sample,))
                halton_mask = torch.zeros(nb_sample, trainer.input_size ** 2, 2, dtype=torch.long)
                for i_h in range(nb_sample):
                    rand_halton = torch.roll(self.basic_halton_mask.clone(), randomize_mask[i_h].item(), 0)
                    halton_mask[i_h] = rand_halton
            else:
                halton_mask = self.basic_halton_mask.clone().unsqueeze(0).expand(nb_sample, trainer.input_size ** 2, 2)

            bar = tqdm(range(self.step), leave=False) if verbose else range(self.step)
            prev_r = 0
            for index in bar:
                # Compute the number of tokens to predict
                ratio = ((index + 1) / self.step)
                r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
                r = int(r * (trainer.input_size ** 2))
                r = max(index + 1, r)

                # U_t: newly released token positions at this step
                _u = halton_mask.clone()[:, prev_r:r]
                U_t = torch.zeros(nb_sample, trainer.input_size, trainer.input_size, dtype=torch.bool)
                for i_mask in range(nb_sample):
                    U_t[i_mask, _u[i_mask, :, 0], _u[i_mask, :, 1]] = True

                # M_t: all released token positions up to this step
                _m = halton_mask.clone()[:, :r]
                M_t = torch.zeros(nb_sample, trainer.input_size, trainer.input_size, dtype=torch.bool)
                for i_mask in range(nb_sample):
                    M_t[i_mask, _m[i_mask, :, 0], _m[i_mask, :, 1]] = True

                # active_mask sent to the transformer only in partial_update mode
                vit_active_mask = None
                if partial_update and index <= self.step // 2: #只在前半部分的step使用partial_update
                    vit_active_mask = U_t.to(trainer.args.device)

                # Choose softmax temperature
                _temp = self.temperature[index] ** self.temp_pow
                if index < self.temp_warmup:
                    _temp *= 0.5  # Reduce temperature during warmup

                if self.w != 0:  # Model prediction with CFG
                    am_cat = (
                        torch.cat([vit_active_mask, vit_active_mask], dim=0)
                        if vit_active_mask is not None else None
                    )
                    with trainer.autocast:
                        logit = trainer.vit(
                            torch.cat([code.clone(), code.clone()], dim=0),
                            torch.cat([labels, labels], dim=0),
                            torch.cat([~drop, drop], dim=0),
                            active_mask=am_cat,
                        )
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    logit = (1 + self.w) * logit_c - self.w * logit_u
                else:
                    with trainer.autocast:
                        logit = trainer.vit(
                            code.clone(), labels, ~drop,
                            active_mask=vit_active_mask,
                        )

                # Compute probabilities using softmax
                prob = torch.softmax(logit * _temp, -1)
                if self.top_k > 0:  # Apply top-k filtering
                    top_k_probs, top_k_indices = torch.topk(prob, self.top_k)
                    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
                    next_token_index = torch.multinomial(top_k_probs.view(-1, self.top_k), num_samples=1)
                    pred_code = top_k_indices.gather(-1, next_token_index.view(nb_sample, trainer.input_size ** 2, 1))
                else:
                    # Sample from the categorical distribution
                    pred_code = torch.distributions.Categorical(probs=prob).sample()

                # Update code with new predictions at U_t positions
                code[U_t] = pred_code.view(nb_sample, trainer.input_size, trainer.input_size)[U_t]

                l_codes.append(pred_code.view(nb_sample, trainer.input_size, trainer.input_size).clone())
                l_U_t.append(U_t.clone().float())
                l_M_t.append(M_t.clone().float())
                prev_r = r

            # Decode the final prediction
            code = torch.clamp(code, 0, trainer.args.codebook_size - 1)
            x = trainer.ae.decode_code(code)
            x = torch.clamp(x, -1, 1)

        trainer.vit.train()  # Restore training mode
        return x, l_codes, l_U_t, l_M_t

    @staticmethod
    def build_halton_mask(input_size, nb_point=10_000):
        """ Generate a halton 'quasi-random' sequence in 2D.
          :param
            input_size -> int: size of the mask, (input_size x input_size).
            nb_point   -> int: number of points to be sample, it should be high to cover the full space.
            h_base     -> torch.LongTensor: seed for the sampling.
          :return:
            mask -> Torch.LongTensor: (input_size x input_size) the mask where each value corresponds to the order of sampling.
        """

        def halton(b, n_sample):
            """Naive Generator function for Halton sequence."""
            n, d = 0, 1
            res = []
            for index in range(n_sample):
                x = d - n
                if x == 1:
                    n = 1
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    n = (b + 1) * y - x
                res.append(n / d)
            return res

        # Sample 2D mask
        data_x = torch.asarray(halton(2, nb_point)).view(-1, 1)
        data_y = torch.asarray(halton(3, nb_point)).view(-1, 1)
        mask = torch.cat([data_x, data_y], dim=1) * input_size
        mask = torch.floor(mask)

        # remove duplicate
        indexes = np.unique(mask.numpy(), return_index=True, axis=0)[1]
        mask = [mask[index].numpy().tolist() for index in sorted(indexes)]
        return torch.LongTensor(np.array(mask))
