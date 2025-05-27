from torch.utils.data import Dataset

class Visium_HES_Dataset(Dataset):
    def __init__(self, slide, coordinates, transform=None):
        """Initialize the Visium_HES_Dataset

        Args:
            slide (OpenSlide): HES slide
            coordinates (DataFrame): visium coordinates with x and y columns
            transform (torchvision.transforms.transforms.Compose, optional): Transform function. Defaults to None.
        """
        self.slide = slide  # HES slide
        self.coordinates = coordinates
        self.x = self.coordinates.x.values
        self.y = self.coordinates.y.values
        self.transform = transform  # Transform function


    def __len__(self):
        """Return the number of coordinates 

        Returns:
            int: Number of coordinates
        """
        return self.coordinates.shape[0]


    def __getitem__(self, idx):
        """Return the image tile at the given index

        Args:
            idx (int): Index of the image tile

        Returns:
            torch.Tensor: Image tile
        """
        # Extract a tile from the slide
        tile = self.slide.read_region(location=(int(self.x[idx]), int(self.y[idx])),
                                       level=0,
                                       size=(448, 448))

        # Convert the tile to RGB
        tile = tile.convert("RGB")

        # Transform the tile using the transform function
        tile = self.transform(tile)

        return tile