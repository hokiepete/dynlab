





class iLES(EulerianDiagnostic2D, RidgeExtractor2D):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...] = None,
        v: np.ndarray[np.ndarray[float, ...], ...] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: tuple[float, float] = None,
        kind: str = 'attacting',
        edge_order: int = 1,
        percentile: float = None,
        force_eigenvectors: bool = False,
        debug: bool = False
    ) -> np.ndarray[np.ndarray[float, ...], ...]:
        if kind.lower() == 'attracting':
            eig_i = 0
        elif kind.lower() == 'repelling':
            eig_i = -1
        else:
            raise ValueError(
                f'kind: {kind}, unrecognized, please use either "attracting" or "repelling"'
            )
        super().compute(x, y, u, v, f, t)

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.rate_field = np.ma.empty([self.ydim, self.xdim])
        self.Xi_max = np.ma.empty([self.ydim, self.xdim, 2])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                eigenValues, eigenVectors = np.linalg.eig(S)
                idx = eigenValues.argsort()
                self.rate_field[i, j] = eigenValues[idx[eig_i]]
                self.Xi_max[i, j, :] = eigenVectors[:, idx[eig_i]]

            else:
                # If the data is masked, then mask the grid point in the output.
                self.rate_field[i, j] = np.ma.masked
                self.Xi_max[i, j, 0] = np.ma.masked
                self.Xi_max[i, j, 1] = np.ma.masked

        # derivatives are no longer needed deleting to be more space efficent and allow larger
        # fields.
        del dudx, dudy, dvdx, dvdy

        if kind == 'attracting':
            self.rate_field = -self.rate_field

        if force_eigenvectors:
            self.Xi_max = force_eigenvectors2D(self.Xi_max)

        self.iles = self.extract_ridges(self.rate_field, self.Xi_max, percentile, edge_order, debug)

        # free up some extra memory if not needed
        if not debug:
            self.Xi_max
        return self.iles
