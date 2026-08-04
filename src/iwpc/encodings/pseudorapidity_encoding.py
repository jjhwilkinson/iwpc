import torch
from torch import Tensor
from iwpc.encodings.encoding_base import Encoding

  class PseudorapidityEncoding(Encoding):                                                                                                                                                                        
      """                                                                                                                                                                                                        
      Encodes a polar angle as pseudorapidity, eta = -ln tan(theta / 2).              
      """                               

    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension
            The number of features to expect
        """
        super().__init__(dimension, dimension)                                                                                                                             
                                                                                                                                                                                                                 
    def _encode(self, x: Tensor) -> Tensor:   
            """
        Parameters
        ----------
        x
            A tensor of dimension (..., dimension)

        Returns
        -------
        Tensor
            Psuedorapidity encoded tensor of dimension (..., dimension)
        """                                                                                                                                                                 
        return -torch.log(torch.tan(x / 2))