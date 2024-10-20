import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

from shapely.geometry import  MultiPoint, Point
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as pgn
from skimage.draw import polygon_perimeter

from dataclasses import dataclass, field
from typing import List, Tuple



@dataclass
class Tissue():

    ncells_x:int
    ncells_y:int
    diamx:float | int
    diamy:float | int
    offset:float | int=50
    rseed:int=16
    
    vor: Voronoi = field(init=False)
    cell_nuclei_pos:np.ndarray = field(init=False)
    cell_regions:np.ndarray = field(init=False)
    cell_vertices:np.ndarray = field(init=False)
    cell_poly_vertices:np.ndarray  = field(init=False)

    def __post_init__(self):
        self.cell_nuclei_pos = self.generate_cell_nuclei_pos() 
        self.vor = Voronoi(self.cell_nuclei_pos)
        self.cell_regions, self.cell_vertices = self.voronoi_finite_polygons_2d(radius=None)
        self.cell_poly_vertices = self.calculate_polygon_vertices()



    def generate_cell_nuclei_pos(self) -> np.ndarray:

        """Generates the positions of cell nuclei."""

        np.random.seed(self.rseed)


        Xpts = np.arange(0, int(self.diamx*self.ncells_x), self.diamx) + self.offset
        Ypts = np.arange(0, int(self.diamy*self.ncells_y), self.diamy) + self.offset

        X2D,Y2D = np.meshgrid(Ypts,Xpts, )
        points = np.column_stack((Y2D.ravel(),X2D.ravel()))


        X2D,Y2D = np.meshgrid(Ypts,Xpts, )
        cell_nuclei_pos = np.column_stack((Y2D.ravel(),X2D.ravel()))

        cell_nuclei_pos = cell_nuclei_pos + np.random.normal(0, 1, cell_nuclei_pos.shape)

        return cell_nuclei_pos
    

    def voronoi_finite_polygons_2d(self, radius=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        source: https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """
        vor=self.vor

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        cell_regions = []
        cell_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges:dict = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                cell_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(cell_vertices))
                cell_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([cell_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region_to_append = np.array(new_region)[np.argsort(angles)]

            # finish
            cell_regions.append(new_region_to_append.tolist())

        return np.asarray(cell_regions, dtype="object"), np.asarray(cell_vertices)
    
    def calculate_polygon_vertices(self)-> np.ndarray:

        pts = MultiPoint([Point(i) for i in self.cell_nuclei_pos])
        mask = pts.convex_hull
        cell_poly_vertices = []
        for region in self.cell_regions:
            polygon = self.cell_vertices[region]
            shape = list(polygon.shape)
            shape[0] += 1
            p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            cell_poly_vertices.append(poly)

        return np.asarray(cell_poly_vertices, dtype="object")


    def show_cells(self) -> None:
 
        for poly in self.cell_poly_vertices:
            plt.fill(*zip(*poly.tolist()), alpha=0.7,  facecolor='lightsalmon',  edgecolor='black', linewidth = 0.1)
            
        plt.plot(self.cell_nuclei_pos[:,0], self.cell_nuclei_pos[:,1], 'ko', markersize=.3)
        plt.title("Tissue simulation")
        plt.xlabel('$\\mu m$')
        plt.ylabel('$\\mu m$')
        plt.show()
        plt.close()


    def create_cellular_lipid_distr_stripes(self,
                                            stripes1:float|int, 
                                            stripes2:float|int, 
                                            stripes1_var:float|int, 
                                            stripes2_var:float|int, 
                                            dimx:int=1000, 
                                            dimy:int=1000, 
                                            resol:int=5) -> np.ndarray:
        
        vert = self.cell_poly_vertices.astype("object")
        
        img_tot = np.zeros((dimx,dimy)) #[int(dimx/10):int(-dimx/10),int(dimx/10):int(-dimy/10)]
        
        total_cells = self.ncells_x * self.ncells_y
        circ_ind = np.arange(-1,total_cells)[::2] - 1
        
        circ_ind_arr = np.zeros(total_cells)

        for i in range(circ_ind_arr.shape[0]):
            for j in range(circ_ind.shape[0]):
                if i == circ_ind[j]:
                    circ_ind_arr[i] = i
            else:
                pass

        for i in range(vert.shape[0]):
            img = np.zeros((dimx,dimy)) #[int(dimx/10):int(-dimx/10),int(dimx/10):int(-dimy/10)]

            if i == circ_ind_arr[i] and i != 0:
                rr, cc = pgn(resol*vert[i][:,0], resol*vert[i][:,1], img.shape)
                img[rr,cc] = stripes1 + np.random.normal(0, stripes1_var)
                img_tot += img
            else:
                rr, cc = pgn(resol*vert[i][:,0], resol*vert[i][:,1], img.shape)

                img[rr,cc] = stripes2 + np.random.normal(0, stripes2_var)
                img_tot += img
        
        return img_tot #[int(dimx/10):int(-dimx/10),int(dimx/10):int(-dimy/10)]