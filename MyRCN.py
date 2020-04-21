import os
import scipy
import numpy as np
from PIL import Image
import networkx as nx
from scipy import signal
from multiprocessing import Pool
from alisuretool.Tools import Tools
# from science_rcn.dilation.dilation import dilate_2d
from scipy.ndimage.morphology import grey_dilation as dilate_2d


class PreProcessing(object):
    """A simplified preprocessing layer implementing Gabor filters and suppression.
    Gabor函数是一个用于边缘提取的线性滤波器。
    """

    def __init__(self, num_orients=16, filter_scale=4., cross_channel_pooling=False):
        self.num_orients = num_orients
        self.filter_scale = filter_scale
        self.cross_channel_pooling = cross_channel_pooling
        self.suppression_masks = self._generate_suppression_masks(filter_scale=filter_scale, num_orients=num_orients)  # 16,13,13
        self.filters = self._get_gabor_filters(21, self.filter_scale, num_orients=self.num_orients, weights=False)  # 16,21,21
        self.pos_filters = self._get_gabor_filters(21, self.filter_scale, num_orients=self.num_orients, weights=True)  # 16,21,21
        pass

    def fwd_infer(self, img, brightness_diff_threshold=40.):
        """Compute bottom-up (forward) inference."""
        filtered = np.zeros((len(self.filters),) + img.shape, dtype=np.float32)
        for i, kern in enumerate(self.filters):
            filtered[i] = signal.fftconvolve(img, kern, mode='same')
        localized = self._local_non_max_suppression(filtered, self.suppression_masks)
        # Threshold and binarize
        localized *= (filtered / brightness_diff_threshold).clip(0, 1)
        localized[localized < 1] = 0

        if self.cross_channel_pooling:
            pooled_channel_weights = [(0, 1), (-1, 1), (1, 1)]
            pooled_channels = [-np.ones_like(sf) for sf in localized]
            for i, pc in enumerate(pooled_channels):
                for channel_offset, factor in pooled_channel_weights:
                    ch = (i + channel_offset) % self.num_orients
                    pos_chan = localized[ch]
                    if factor != 1:
                        pos_chan[pos_chan > 0] *= factor
                    np.maximum(pc, pos_chan, pc)
            bu_msg = np.array(pooled_channels)
        else:
            bu_msg = localized
        # Setting background to -1
        bu_msg[bu_msg == 0] = -1.
        return bu_msg

    @staticmethod
    def _generate_suppression_masks(filter_scale=4., num_orients=16):
        """
        Generate the masks for oriented non-max suppression at the given filter_scale.
        See Preproc for parameters and returns.
        """
        size = 2 * int(np.ceil(filter_scale * np.sqrt(2))) + 1
        cx, cy = size // 2, size // 2
        filter_masks = np.zeros((num_orients, size, size), np.float32)
        # Compute for orientations [0, pi), then flip for [pi, 2*pi)
        for i, angle in enumerate(np.linspace(0., np.pi, num_orients // 2, endpoint=False)):
            x, y = np.cos(angle), np.sin(angle)
            for r in range(1, int(np.sqrt(2) * size / 2)):
                dx, dy = round(r * x), round(r * y)
                if abs(dx) > cx or abs(dy) > cy:
                    continue
                filter_masks[i, int(cy + dy), int(cx + dx)] = 1
                filter_masks[i, int(cy - dy), int(cx - dx)] = 1
        filter_masks[num_orients // 2:] = filter_masks[:num_orients // 2]
        return filter_masks

    @staticmethod
    def _get_gabor_filters(size=21, filter_scale=4., num_orients=16, weights=False):
        """Get Gabor filter bank. See Preproc for parameters and returns.
        now = self.filters[1]
        now_sub_min = now - now.min()
        now_div_max = now_sub_min / now_sub_min.max()
        im = np.asarray(now_div_max * 255, dtype=np.uint8)
        Image.fromarray(im).show()
        """

        def _get_sparse_gaussian():
            size = 2 * np.ceil(np.sqrt(2.) * filter_scale) + 1
            alt = np.zeros((int(size), int(size)), np.float32)
            alt[int(size // 2), int(size // 2)] = 1
            gaussian = scipy.ndimage.filters.gaussian_filter(alt, filter_scale / np.sqrt(2.), mode='constant')
            gaussian[gaussian < 0.05 * gaussian.max()] = 0
            return gaussian

        gaussian = _get_sparse_gaussian()
        filters = []
        for angle in np.linspace(0., 2 * np.pi, num_orients, endpoint=False):
            acts = np.zeros((size, size), np.float32)
            x, y = np.cos(angle) * filter_scale, np.sin(angle) * filter_scale
            acts[int(size / 2 + y), int(size / 2 + x)] = 1.
            acts[int(size / 2 - y), int(size / 2 - x)] = -1.
            _filter = signal.fftconvolve(acts, gaussian, mode='same')
            _filter = _filter / np.abs(_filter).sum()
            if weights:
                _filter = np.abs(_filter)
            filters.append(_filter)
        return filters

    @staticmethod
    def _local_non_max_suppression(filtered, suppression_masks):
        """
        Apply oriented non-max suppression to the filters, so that only a single 
        orientated edge is active at a pixel. See Preproc for additional parameters.

        Parameters
        ----------
        filtered : numpy.ndarray
            Output of filtering the input image with the filter bank.
            Shape is (num feats, rows, columns).

        Returns
        -------
        localized : numpy.ndarray
            Result of oriented non-max suppression.
        """
        localized = np.zeros_like(filtered)
        cross_orient_max = filtered.max(0)  # 每个位置的最大值
        filtered[filtered < 0] = 0
        for i, (layer, suppress_mask) in enumerate(zip(filtered, suppression_masks)):
            competitor_maxs = scipy.ndimage.maximum_filter(layer, footprint=suppress_mask, mode='nearest')
            localized[i] = competitor_maxs <= layer  # 极大值的地方
        localized[filtered < cross_orient_max] = 0  # 所有特征通道上最大值在该特征图上的最大值
        return localized

    pass


class Learning(object):

    """Learn a two-layer RCN model."""

    @staticmethod
    def sparsify(bu_msg, suppress_radius=3):
        """Make a sparse representation of the edges by greedily selecting features from the 
        output of preprocessing layer and suppressing overlapping activations.
        稀疏化表示，并通过贪婪选择稀疏
        """
        frcs = []
        img = bu_msg.max(0) > 0
        while True:
            r, c = np.unravel_index(img.argmax(), img.shape)
            if not img[r, c]:
                break
            frcs.append((bu_msg[:, r, c].argmax(), r, c))
            img[r - suppress_radius:r + suppress_radius + 1, c - suppress_radius:c + suppress_radius + 1] = False
            pass
        return np.array(frcs)

    @classmethod
    def learn_laterals(cls, frcs, bu_msg, perturb_factor, use_adjaceny_graph=False):
        """Given the sparse representation of each training example,
        learn perturbation laterals. See train_image for parameters and returns.
        """
        if use_adjaceny_graph:
            graph = cls.make_adjacency_graph(frcs, bu_msg)
            graph = cls.adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)
        else:
            graph = nx.Graph()
            graph.add_nodes_from(range(frcs.shape[0]))
            pass

        graph = cls.add_under_constraint_edges(frcs, graph, perturb_factor=perturb_factor)
        graph = cls.adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)

        edge_factors = np.array([(edge_source, edge_target, edge_attr['perturb_radius'])
                                 for edge_source, edge_target, edge_attr in graph.edges(data=True)])
        return graph, edge_factors

    @staticmethod
    def make_adjacency_graph(frcs, bu_msg, max_dist=3):
        """Make a graph based on contour adjacency."""
        preproc_pos = np.transpose(np.nonzero(bu_msg > 0))[:, 1:]
        preproc_tree = scipy.spatial.cKDTree(preproc_pos)
        # Assign each preproc to the closest F1
        f1_bus_tree = scipy.spatial.cKDTree(frcs[:, 1:])
        _, preproc_to_f1 = f1_bus_tree.query(preproc_pos, k=1)
        # Add edges
        preproc_pairs = np.array(list(preproc_tree.query_pairs(r=max_dist, p=1)))
        f1_edges = np.array(list({(x, y) for x, y in preproc_to_f1[preproc_pairs] if x != y}))

        graph = nx.Graph()
        graph.add_nodes_from(range(frcs.shape[0]))
        graph.add_edges_from(f1_edges)
        return graph

    @staticmethod
    def add_under_constraint_edges(frcs, graph, perturb_factor=2., max_cxn_length=100, tolerance=4):
        """Examines all pairs of variables and greedily adds pairwise constraints until the pool 
        flexibility matches the desired amount of flexibility specified by perturb_factor and tolerance."""
        graph = graph.copy()
        f1_bus_tree = scipy.spatial.cKDTree(frcs[:, 1:])

        close_pairs = np.array(list(f1_bus_tree.query_pairs(r=max_cxn_length)))
        dists = [scipy.spatial.distance.euclidean(frcs[x, 1:], frcs[y, 1:]) for x, y in close_pairs]

        for close_pairs_idx in np.argsort(dists):
            source, target = close_pairs[close_pairs_idx]
            dist = dists[close_pairs_idx]

            try:
                perturb_dist = nx.shortest_path_length(graph, source, target, 'perturb_radius')
            except nx.NetworkXNoPath:
                perturb_dist = np.inf

            target_perturb_dist = dist / float(perturb_factor)
            actual_perturb_dist = max(0, np.ceil(target_perturb_dist))
            if perturb_dist >= target_perturb_dist * tolerance:
                graph.add_edge(source, target, perturb_radius=int(actual_perturb_dist))
        return graph

    @staticmethod
    def adjust_edge_perturb_radii(frcs, graph, perturb_factor=2):
        """Returns a new graph where the 'perturb_radius' has been adjusted to account for 
        rounding errors. See train_image for parameters and returns.
        """
        graph = graph.copy()

        total_rounding_error = 0
        for n1, n2 in nx.edge_dfs(graph):
            desired_radius = scipy.spatial.distance.euclidean(frcs[n1, 1:], frcs[n2, 1:]) / perturb_factor

            upper = int(np.ceil(desired_radius))
            lower = int(np.floor(desired_radius))
            round_up_error = total_rounding_error + upper - desired_radius
            round_down_error = total_rounding_error + lower - desired_radius
            if abs(round_up_error) < abs(round_down_error):
                graph[n1][n2]['perturb_radius'] = upper
                total_rounding_error = round_up_error
            else:
                graph[n1][n2]['perturb_radius'] = lower
                total_rounding_error = round_down_error
        return graph

    pass


class Inference(object):

    @classmethod
    def forward_pass(cls, frcs, bu_msg, graph, pool_shape):
        """Forward pass inference using a tree-approximation (cf. Sec S4.2)."""
        height, width = bu_msg.shape[-2:]
        vps, hps = pool_shape  # Vertical and horizontal pool shapes

        def _pool_slice(f, r, c):
            assert (r - vps // 2 >= 0 and r + vps - vps // 2 < height and
                    c - hps // 2 >= 0 and c + hps - hps // 2 < width), \
                "Some pools are out of the image boundaries. " \
                "Consider increase image padding or reduce pool shapes."
            return np.s_[f, r - vps // 2: r + vps - vps // 2, c - hps // 2: c + hps - hps // 2]

        # Find a schedule to compute the max marginal for the most constrained tree
        tree_schedule = cls.get_tree_schedule(frcs, graph)

        # If we're sending a message out from x to y, it means x has received all incoming messages
        incoming_msgs = {}
        for source, target, perturb_radius in tree_schedule:
            msg_in = bu_msg[_pool_slice(*frcs[source])]
            if source in incoming_msgs:
                msg_in = msg_in + incoming_msgs[source]
                del incoming_msgs[source]
            msg_in = dilate_2d(msg_in, (2 * perturb_radius + 1, 2 * perturb_radius + 1))
            if target in incoming_msgs:
                incoming_msgs[target] += msg_in
            else:
                incoming_msgs[target] = msg_in
        fp_score = np.max(incoming_msgs[tree_schedule[-1, 1]] + bu_msg[_pool_slice(*frcs[tree_schedule[-1, 1]])])
        return fp_score

    @staticmethod
    def get_tree_schedule(frcs, graph):
        """Find the most constrained tree in the graph and returns which messages to compute it. 
        This is the minimum spanning tree of the perturb_radius edge attribute."""
        min_tree = nx.minimum_spanning_tree(graph, 'perturb_radius')
        return np.array([(target, source, graph[source][target]['perturb_radius'])
                         for source, target in nx.dfs_edges(min_tree)])[::-1]

    pass


class LoopyBPInference(object):

    """Max-product loopy belief propagation for a two-level RCN model (cf. Sec S4.4)."""

    def __init__(self, bu_msg, frcs, edge_factors, pool_shape, preproc_layer, n_iters=300, damping=1.0, tol=1e-5):
        """
        Parameters
        ----------
        bu_msg : numpy.array of float
            Bottom up messages from preprocessing layer, in the following format:
            (feature idx, row, col).
        frcs : np.ndarray of np.int
            Nx3 array of (feature idx, row, column), where each row represents a
            single pool center.
        edge_factors : numpy.ndarray of numpy.int
            Nx3 array of (source pool index, target pool index, perturb_radius), where
            each row is a pairwise constraints on a pair of pool choices.
        pool_shape : (int, int)
            Vertical and horizontal pool shapes.
        preproc_layer : Preproc
            Pre-processing layer. See preproc.py.
        n_iters : int
            Maximum number of loopy BP iterations.
        damping : float
            Damping parameter for loopy BP.
        tol : float
            Tolerance to determine loopy BP convergence.
        """
        self.n_feats, self.n_rows, self.n_cols = bu_msg.shape
        self.n_pools, self.n_factors = frcs.shape[0], edge_factors.shape[0]
        self.vps, self.hps = pool_shape
        self.frcs = frcs
        self.bu_msg = bu_msg
        self.edge_factors = edge_factors
        self.preproc_layer = preproc_layer
        self.n_iters = n_iters
        self.damping = damping
        self.tol = tol

        # Check inputs
        if (np.array([0, self.vps // 2, self.hps // 2]) > frcs.min(0)).any():
            raise Exception("Some frcs are too small for the provided pool shape")
        if (frcs.max(0)>=np.array([self.n_feats, self.n_rows-((self.vps-1)//2), self.n_cols-((self.hps-1)//2)])).any():
            raise Exception("Some frcs are too big for the provided pool shape and/or `bu_msg`")
        if (edge_factors[:, :2].min(0) < np.array([0, 0])).any():
            raise Exception("Some variable index in `edge_factors` is negative")
        if (edge_factors[:, :2].max(0) >= np.array([self.n_pools, self.n_pools])).any():
            raise Exception("Some index in `edge_factors` exceeds the number of vars")
        if (edge_factors[:, 0] == edge_factors[:, 1]).any():
            raise Exception("Some factor connects a variable to itself")
        if not issubclass(edge_factors.dtype.type, np.integer):
            raise Exception("Factors should be an integer numpy array")

        # Initialize message
        self._reset_messages()
        self.unary_messages = np.zeros((self.n_pools, self.vps, self.hps))
        bu_msg_pert = self.bu_msg + 0.01 * (2 * np.random.rand(*bu_msg.shape) - 1)
        for i, (f, r, c) in enumerate(self.frcs):
            rstart = r - self.vps // 2
            cstart = c - self.hps // 2
            self.unary_messages[i] = bu_msg_pert[f, rstart:rstart + self.vps, cstart:cstart + self.hps]
            pass
        pass

    def _reset_messages(self):
        """Set all lateral messages to zero."""
        self.lat_messages = np.zeros((2, self.n_factors, self.vps, self.hps))

    @staticmethod
    def compute_1pl_message(in_mess, pert_radius):
        """Compute the outgoing message of a lateral factor given the
        perturbation radius and input message.

        Parameters
        ----------
        in_mess : numpy.array
            Input BP messages to the factor. Each message has shape vps x hps.
        pert_radius : int
            Perturbation radius corresponding to the factor.

        Returns
        -------
        out_mess : numpy.array
            Output BP message (at the opposite end of the factor from the input message).
            Shape is (vps, hps).
        """
        pert_diameter = 2 * pert_radius + 1
        out_mess = dilate_2d(in_mess, (pert_diameter, pert_diameter))
        return out_mess - out_mess.max()

    def new_messages(self):
        """Compute updated set of lateral messages (in both directions).

        Returns
        -------
        new_lat_messages : numpy.array
            Updated set of lateral messages. Shape is (2, n_factors, vps x hps).
        """
        # Compute beliefs
        beliefs = self.unary_messages.copy()
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            beliefs[var_j] += self.lat_messages[0, f]
            beliefs[var_i] += self.lat_messages[1, f]

        # Compute outgoing messages
        new_lat_messages = np.zeros_like(self.lat_messages)
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            new_lat_messages[0, f] = self.compute_1pl_message(beliefs[var_i] - self.lat_messages[1, f], pert_radius)
            new_lat_messages[1, f] = self.compute_1pl_message(beliefs[var_j] - self.lat_messages[0, f], pert_radius)
        return new_lat_messages

    def bwd_pass(self):
        """Perform max-product loopy BP inference and decode the max-marginals.

        Returns
        -------
        score : float
            The score of the backtraced solution, adjusted for filter overlapping.
        """
        self._reset_messages()
        self.infer_pbp()  # Loopy BP with parallel updates
        assignments, backtrace_positions, score = self.decode()  # Decode the max-marginals
        if not self.laterals_are_satisfied(assignments):  # Check constraints are satisfied
            Tools.print("Lateral constraints not satisfied. Try increasing the number of iterations.")
            score = -np.inf
        return score

    def infer_pbp(self):
        """Parallel loopy BP message passing, modifying state of `lat_messages`. See bwd_pass() for parameters."""
        for it in range(self.n_iters):
            new_lat_messages = self.new_messages()
            delta = new_lat_messages - self.lat_messages
            self.lat_messages += self.damping * delta
            if np.abs(delta).max() < self.tol:
                Tools.print("Parallel loopy BP converged in {} iterations".format(it))
                return
        Tools.print("Parallel loopy BP didn't converge in {} iterations".format(self.n_iters))
        pass

    def decode(self):
        """Find pool assignments by decoding the max-marginal messages.

        Returns
        -------
        assignments : 2D numpy.ndarray of int
            Each row is the row and column assignments for each pool.
        backtrace_positions : 3D numpy.ndarray of int
            Sparse top-down activations in the form of (f,r,c).
        score : float
            Sum of log-likelihoods collected by the decoded pool assignments.
        """
        # Compute beliefs
        beliefs = self.unary_messages.copy()
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            beliefs[var_j] += self.lat_messages[0, f]
            beliefs[var_i] += self.lat_messages[1, f]

        assignments = np.zeros((self.n_pools, 2), dtype=np.int)
        backtrace = np.zeros((self.n_feats, self.n_rows, self.n_cols))
        for i, (f, r, c) in enumerate(self.frcs):
            r_max, c_max = np.where(beliefs[i] == beliefs[i].max())
            choice = np.random.randint(len(r_max))
            assignments[i] = np.array([r_max[choice], c_max[choice]])
            rstart = r - self.vps // 2
            cstart = c - self.hps // 2
            backtrace[f, rstart + assignments[i, 0], cstart + assignments[i, 1]] = 1
        backtrace_positions = np.transpose(np.nonzero(backtrace))
        score = self.recount(backtrace_positions, self.bu_msg, self.preproc_layer.pos_filters)
        return assignments, backtrace_positions, score

    def laterals_are_satisfied(self, assignments):
        """Check whether pool assignments satisfy all lateral constraints.

        Parameters
        ----------
        assignments : 2D numpy.ndarray of int
            Row and column assignments for each pool.

        Returns
        -------
        satisfied : bool
            Whether the pool assignments satisfy all lateral constraints.
        """
        satisfied = True
        for f, (var_i, var_j, pert_radius) in enumerate(self.edge_factors):
            rdist, cdist = np.abs(assignments[var_i] - assignments[var_j])
            if not (rdist <= pert_radius and cdist <= pert_radius):
                satisfied = False
                break
        return satisfied

    @staticmethod
    def recount(backtrace_positions, bu_msg, filters):
        """
        Post-processing step to prevent overcounting of log-likelihoods (cf. Sec S8.2).

        Parameters
        ----------
        backtrace_positions : 3D numpy.ndarray of int
            Sparse top-down activations in the format of (f,r,c).
        bu_msg : 3D numpy.ndarray of int
            Bottom-up messages after the pre-processing layer.
        filters : [2D numpy.ndarray of float]
            Filter bank used in the pre-processing layer.

        Returns
        -------
        normalized_score : float
            Score normalized by taking filter overlaps into account.
        """
        height, width = bu_msg.shape[-2:]
        f_h, f_w = filters[0].shape
        layers = np.zeros((len(backtrace_positions), height, width))
        fo_h, fo_w = f_h // 2, f_w // 2
        from_r, to_r = (np.maximum(0, backtrace_positions[:, 1] - fo_h),
                        np.minimum(height, backtrace_positions[:, 1] - fo_h + f_h))
        from_c, to_c = (np.maximum(0, backtrace_positions[:, 2] - fo_w),
                        np.minimum(width, backtrace_positions[:, 2] - fo_w + f_w))
        from_fr, to_fr = (np.maximum(0, fo_h - backtrace_positions[:, 1]),
                          np.minimum(f_h, height - backtrace_positions[:, 1] + fo_h))
        from_fc, to_fc = (np.maximum(0, fo_w - backtrace_positions[:, 2]),
                          np.minimum(f_w, width - backtrace_positions[:, 2] + fo_w))

        if not np.all(to_r - from_r == to_fr - from_fr):
            raise Exception("Numbers of rows of filter and image patches "
                            "({}, {}) do not agree".format(to_r - from_r, to_fr - from_fr))
        if not np.all(to_c - from_c == to_fc - from_fc):
            raise Exception("Numbers of columns of filter and image patches "
                            "({}, {}) do not agree".format(to_c - from_c, to_fc - from_fc))

        # Normalize activations by taking into account filter overlaps
        weight_sum = np.zeros((height, width))
        for i, (f, r, c) in enumerate(backtrace_positions):
            # Convolve sparse top-down activations with filters
            filt = filters[f][from_fr[i]:to_fr[i], from_fc[i]:to_fc[i]]

            weight_sum[from_r[i]:to_r[i], from_c[i]:to_c[i]] += filt
            layers[i, from_r[i]:to_r[i], from_c[i]:to_c[i]] = filt ** 2 * bu_msg[f, r, c] / (1e-9 + filt.sum())
        normalized_score = (layers.sum(0) / (1e-9 + weight_sum)).sum()
        return normalized_score

    pass


class RCN(object):

    def __init__(self, train_size=20, test_size=20, full_test_set=False,
                 pool_shape=(25, 25), perturb_factor=2., data_dir='.\\data\\MNIST'):
        self.train_size = train_size
        self.test_size = test_size
        self.full_test_set = full_test_set
        self.pool_shape = pool_shape
        self.perturb_factor = perturb_factor
        self.data_dir = data_dir

        self.train_data, self.train_label, self.test_data, self.test_label = self.get_data()
        pass

    def get_data(self, seed=5):
        np.random.seed(seed)
        train_data, train_label = self._load_data(os.path.join(self.data_dir, 'training'),
                                                  num_per_class=self.train_size//10)
        test_data, test_label = self._load_data(os.path.join(self.data_dir, 'testing'),
                                                num_per_class=None if self.full_test_set else self.test_size//10)
        return train_data, train_label, test_data, test_label

    @staticmethod
    def _load_data(image_dir, num_per_class):
        loaded_data, loaded_label = [], []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            samples = sorted(os.listdir(cat_path))
            if num_per_class is not None:
                samples = np.random.choice(samples, num_per_class)

            for sample in samples:
                image_arr = np.array(Image.open(os.path.join(cat_path, sample)).resize((112, 112)))
                image_arr = np.pad(image_arr, tuple([(p, p) for p in (44, 44)]), mode='constant', constant_values=0)
                loaded_data.append(image_arr)
                loaded_label.append(int(category))
                pass
            pass
        return loaded_data, loaded_label

    def run_experiment(self):
        all_model_factors = self.train()
        test_results = self.test(all_model_factors)
        Tools.print("Total test acc={}".format(sum([result[0] for result in test_results]) / len(test_results)))
        pass

    def train(self):
        Tools.print("Training on {} images...".format(len(self.train_data)))
        train_results = [self._train_image(data, perturb_factor=self.perturb_factor) for data in self.train_data]
        return list(zip(*train_results))

    def test(self, model_factors):
        Tools.print("Testing on {} images...".format(len(self.test_data)))
        test_results = []
        for index, (data, label) in enumerate(zip(self.test_data, self.test_label)):
            test_result = self._test_image(data, model_factors=model_factors, pool_shape=self.pool_shape)
            result_label = self.train_label[test_result[0]]
            test_results.append([label == result_label, label, result_label, test_result[0], test_result[1]])
            Tools.print("{} ok={} label={} pred={} result={}".format(index, label == result_label,
                                                                     label, result_label, test_result))
            pass
        return test_results

    @staticmethod
    def _train_image(img, perturb_factor=2.):
        bu_msg = PreProcessing().fwd_infer(img)  # Pre-processing layer (4.2.1)
        frcs = Learning.sparsify(bu_msg)  # Sparsification (5.1.1)
        graph, edge_factors = Learning.learn_laterals(frcs, bu_msg, perturb_factor)  # Lateral learning (5.2)
        return frcs, edge_factors, graph

    @staticmethod
    def _test_image(img, model_factors, pool_shape=(25, 25), num_candidates=20, n_iters=300, damping=1.0):
        # Get bottom-up messages from the pre-processing layer
        pre_processing_layer = PreProcessing(cross_channel_pooling=True)
        bu_msg = pre_processing_layer.fwd_infer(img)

        # Forward pass inference
        fp_scores = np.zeros(len(model_factors[0]))
        for i, (frcs, _, graph) in enumerate(zip(*model_factors)):
            fp_scores[i] = Inference.forward_pass(frcs, bu_msg, graph, pool_shape)
        top_candidates = np.argsort(fp_scores)[-num_candidates:]

        # Backward pass inference
        winner_idx, winner_score = (-1, -np.inf)  # (training feature idx, score)
        for idx in top_candidates:
            frcs, edge_factors = model_factors[0][idx], model_factors[1][idx]
            score = LoopyBPInference(bu_msg, frcs, edge_factors, pool_shape,
                                     pre_processing_layer, n_iters, damping=damping).bwd_pass()
            if score >= winner_score:
                winner_idx, winner_score = (idx, score)
            pass
        return winner_idx, winner_score

    pass


if __name__ == '__main__':
    _train_size, _test_size, _full_test_set = 10, 10, False
    _pool_shape, _perturb_factor = 25, 2.
    rcn = RCN(train_size=_train_size, test_size=_test_size, full_test_set=_full_test_set,
              pool_shape=(_pool_shape, _pool_shape), perturb_factor=_perturb_factor)
    rcn.run_experiment()
