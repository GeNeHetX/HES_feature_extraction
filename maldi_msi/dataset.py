from torch.utils.data import Dataset

class MALDI_HES_Dataset(Dataset):
    def __init__(self, slide, pixels, transform=None):
        """Initialize the MALDI_HES_Dataset

        Args:
            slide (OpenSlide): HES slide
            pixels (DataFrame): MALDI pixels with x_warped and y_warped columns
            transform (torchvision.transforms.transforms.Compose, optional): Transform function. Defaults to None.
        """
        self.slide = slide  # HES slide
        self.pixels = pixels
        self.x = self.pixels.x_warped.round().astype(int)
        self.y = self.pixels.y_warped.round().astype(int)
        self.transform = transform  # Transform function


    def __len__(self):
        """Return the number of pixels

        Returns:
            int: Number of pixels
        """
        return self.pixels.shape[0]


    def __getitem__(self, idx):
        """Return the image tile at the given index

        Args:
            idx (int): Index of the image tile

        Returns:
            torch.Tensor: Image tile
        """
        # Extract a tile from the slide
        tile = self.slide.read_region(location=(self.x[idx], self.y[idx]),
                                      level=0,
                                      size=(448, 448))

        # Convert the tile to RGB
        tile = tile.convert("RGB")

        # Transform the tile using the transform function
        tile = self.transform(tile)

        return tile