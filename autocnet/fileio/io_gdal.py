import os

import numpy
from osgeo import gdal
from osgeo import osr

from autocnet.fileio import extract_metadata

NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}

GDAL2NP_CONVERSION = {}

for k, v in iter(NP2GDAL_CONVERSION.items()):
    GDAL2NP_CONVERSION[v] = k

GDAL2NP_CONVERSION[1] = 'int8'

class GeoDataset(object):
    """
    Geospatial dataset object that represents.

    Parameters
    ----------
    file_name : str
                The name of the input image, including its full path.

    Attributes
    ----------

    base_name : str
                The base name of the input image, extracted from the full path.

    geotransform : object
                   Represents the geotransform reference OGR object.

    geospatial_coordinate_system : object
                                   Represents the geospatial coordinate system OSR object.

    latlon_extent : list
                    of two tuples to describe that latitide/longitude boundaries. 
                    This list is in the form [(lowerlat, lowerlon), (upperlat, upperlon)].

    pixel_width : float
                  The width of the image pixels (i.e. displacement in the x-direction).

    pixel_height : float
                   The height of the image pixels (i.e. displacement in the y-direction).

    spatial_reference : object
                        Represents the OSR spatial reference system OSR object.

    standard_parallels : list
                         of the standard parallels used by the map projection.

    unit_type : str
                Name of the unit used by the raster, e.g. 'm' or 'ft'.

    x_rotation : float
                The geotransform coefficient that represents the rotation about the x-axis.

    xy_extent : list
                of two tuples to describe the x/y boundaries. 
                This list is in the form [(minx, miny), (maxx, maxy)].

    y_rotation : float
                The geotransform coefficient that represents the rotation about the y-axis.

    """
    def __init__(self, file_name):
        """
        Initialization method to set the file name and open the file using GDAL.

        Parameters
        ----------
        file_name : str
                   The file name to set and open.

        """
        self.file_name = file_name
        self.dataset = gdal.Open(file_name)
    
    def __repr__(self):
        return os.path.basename(self.file_name)

    @property
    def base_name(self):
        """
        Gets the base name of the file (without the full directory path).

        Returns
        -------
        _base_name : str
                     The base file name.

        """
        if not getattr(self, '_base_name', None):
            self._base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        return self._base_name

    @property
    def geotransform(self):
        """
        Gets an array of size 6 containing the affine transformation coefficients for transforming
        from raw sample/line to projected x/y.

        xproj = geotransform[0] + sample * geotransform[1] + line * geotransform[2]
        yproj = geotransform[3] + sample * geotransform[4] + line * geotransform[5]

        Returns
        -------
        _geotransform : array
                        of transformation coefficients.

        """
        if not getattr(self, '_geotransform', None):
            self._geotransform = self.dataset.GetGeoTransform()
        return self._geotransform

    @property
    def standard_parallels(self):
        """
        Gets the list of standard parallels found in the metadata using the spatial reference for
        this GeoDataset.

        Returns
        -------
        _standard_parallels : list
                              of standard parallels.

        """
        if not getattr(self, '_standard_parallels', None):
            self._standard_parallels = extract_metadata.get_standard_parallels(self.spatial_reference)
        return self._standard_parallels

    @property
    def unit_type(self):
        """
        Gets the type of units the raster data is stored in. For example, this might be meters,
        kilometers, feet, etc.

        Returns
        -------
        _unit_type : str
                     The units for this data set.

        """
        if not getattr(self, '_unit_type', None):
            self._unit_type = self.dataset.GetRasterBand(1).GetUnitType()
        return self._unit_type

    @property
    def spatial_reference(self):
        """
        Gets the spatial reference system (SRS) and sets the geospatial coordinate system (GCS).

        Returns
        -------
        _srs : object
               The spatial reference system. 
        
        """
        if not getattr(self, '_srs', None):
            self._srs = osr.SpatialReference()
            self._srs.ImportFromWkt(self.dataset.GetProjection())
            try:
                self._srs.MorphToESRI()
                self._srs.MorphFromESRI()
            except: pass #pragma: no cover

            #Setup the GCS
            self._gcs = self._srs.CloneGeogCS()
        return self._srs

    @property
    def geospatial_coordinate_system(self):
        """
        Gets the geospatial coordinate system (GCS).

        Returns
        -------
        _gcs : object
               The geospatial coordinate system. 
        
        """
        if not getattr(self, '_gcs', None):
            self._gcs = self.spatial_reference.CloneGeogCS()
        return self._gcs

    @property
    def latlon_extent(self):
        """
        Gets the size two list of tuples containing the latitide/longitude boundaries. 
        This list is in the form [(lowerlat, lowerlon), (upperlat, upperlon)].

        Returns
        -------
        _latlon_extent : list
                         [(lowerlat, lowerlon), (upperlat, upperlon)]
        
        """
        if not getattr(self, '_latlon_extent', None):
            xy_extent = self.xy_extent
            lowerlat, lowerlon = self.pixel_to_latlon(xy_extent[0][0], xy_extent[0][1])
            upperlat, upperlon = self.pixel_to_latlon(xy_extent[1][0], xy_extent[1][1])
            self._latlon_extent = [(lowerlat, lowerlon), (upperlat, upperlon)]
        return self._latlon_extent

    @property
    def xy_extent(self):
        """
        Gets the size two list of tuples containing the sample/line boundaries. 
        The first value is the upper left corner of the upper left pixel and 
        the second value is the lower right corner of the lower right pixel. 
        This list is in the form [(minx, miny), (maxx, maxy)].

        Returns
        -------
        _xy_extent : list
                     [(minx, miny), (maxx, maxy)]
        
        """
        if not getattr(self, '_xy_extent', None):
            geotransform = self.geotransform
            minx = geotransform[0]
            maxy = geotransform[3]

            maxx = minx + geotransform[1] * self.dataset.RasterXSize
            miny = maxy + geotransform[5] * self.dataset.RasterYSize

            self._xy_extent = [(minx, miny), (maxx, maxy)]

        return self._xy_extent

    @property
    def pixel_width(self):
        """
        Get the width of the pixels in the input image (i.e. the displacement in the x-direction).
        Note: This is the second value geotransform array.

        Returns
        -------
        _pixel_width : float
                       The width of each pixel.
        
        """
        if not getattr(self, '_pixel_width', None):
            self._pixel_width = self.geotransform[1]
        return self._pixel_width

    @property
    def pixel_height(self):
        """
        Get the height of the pixels in the input image (i.e the displacement in the y-direction).
        Note: This is the sixth (last) value geotransform array.

        Returns
        -------
        _pixel_height : float
                        The height of each pixel.
        
        """
        if not getattr(self, '_pixel_height', None):
            self._pixel_height = self.geotransform[5]
        return self._pixel_height

    @property
    def x_rotation(self):
        """
        Get the geotransform rotation about the x-axis.
        Note: This is the third value geotransform array.

        Returns
        -------
        _x_rotation : float
                     The geotransform coefficient representing rotation about the x-axis.
        
        """
        if not getattr(self, '_x_rotation', None):
            self._x_rotation = self.geotransform[2]
        return self._x_rotation

    @property
    def y_rotation(self):
        """
        Get the geotransform rotation about the y-axis.
        Note: This is the fifth value geotransform array.

        Returns
        -------
        _y_rotation : float
                     The geotransform coefficient representing rotation about the y-axis.
        
        """
        if not getattr(self, '_y_rotation', None):
            self._y_rotation = self.geotransform[4]
        return self._y_rotation

    @property
    def coordinate_transformation(self):
        """
        Gets the coordinate transformation from the spatial reference system to the geospatial 
        coordinate system.

        Returns
        -------
        _ct : object
              The coordinate transformation. 
        
        """
        if not getattr(self, '_ct', None):
            self._ct = osr.CoordinateTransformation(self.spatial_reference,
                                                    self.geospatial_coordinate_system)
        return self._ct

    @property
    def inverse_coordinate_transformation(self):
        """
        Gets the coordinate transformation from the geospatial coordinate system to the spatial 
        reference system.

        Returns
        -------
        _ict : object
               The inverse coordinate transformation.
        
        """
        if not getattr(self, '_ict', None):
                       self._ict = osr.CoordinateTransformation(self.geospatial_coordinate_system,
                                                                self.spatial_reference)
        return self._ict

    @property
    def no_data_value(self, band=1):
        """
        Gets the no data value for the given band. This is used to indicate pixels that are not valid.

        Parameters
        ----------
        band : int
               The one-based index of the band. Default band=1.

        Returns
        -------
        _no_data_value : float
                         Special value used to indicate invalid pixels.
        
        """
        if not getattr(self, '_no_data_value', None):
            self._no_data_value = self.dataset.GetRasterBand(band).GetNoDataValue()
        return self._no_data_value

    @property
    def scale(self):
        """
        Gets the name and value of the linear projection units of the spatial reference system. 
        To transform a linear distance to meters, multiply by this value.
        If no units are available ("Meters", 1) will be returned.

        Returns
        -------
        _scale : tuple
                 A string/float tuple of the form (unit name, value)
                 
        """
        if not getattr(self, '_scale', None):
            unitname = self.spatial_reference.GetLinearUnitsName()
            value = self.spatial_reference.GetLinearUnits()
            self._scale = (unitname, value)
        return self._scale

    @property
    def spheroid(self):
        """
        Gets the spheroid found in the metadata using the spatial reference system. 

        Returns
        -------
        _spheroid : tuple
                    (semi-major, semi-minor, inverse flattening)
        
        """
        if not getattr(self, '_spheroid', None):
            self._spheroid = extract_metadata.get_spheroid(self.spatial_reference)
        return self._spheroid

    @property
    def raster_size(self):
        """
        Gets the dimensions of the raster, i.e. (number of samples, number of lines).

        Returns
        -------
        _raster_size : tuple
                       (x size, y size)
        
        """
        if not getattr(self, '_raster_size', None):
            self._raster_size = (self.dataset.RasterXSize, self.dataset.RasterYSize)
        return self._raster_size

    @property
    def central_meridian(self):
        """
        Gets the central meridian from the metadata.

        Returns
        -------
        _central_meridian : float

        """
        if not getattr(self, '_central_meridian', None):
            self._central_meridian = extract_metadata.get_central_meridian(self.spatial_reference)
        return self._central_meridian

    def pixel_to_latlon(self, x, y):
        """
        Convert from pixel space (i.e. sample/line) to lat/lon space.

        Parameters
        ----------
        x : float
            x-coordinate to be transformed.
        y : float
            y-coordinate to be transformed.

        Returns
        -------
        lat, lon : tuple
                   (Latitude, Longitude) corresponding to the given (x,y).
        
        """
        geotransform = self.geotransform
        x = geotransform[0] + (x * geotransform[1]) + (y * geotransform[2])
        y = geotransform[3] + (x * geotransform[4]) + (y * geotransform[5])
        lon, lat, _ = self.coordinate_transformation.TransformPoint(x, y)

        return lat, lon

    def latlon_to_pixel(self, lat, lon):
        """
        Convert from lat/lon space to pixel space (i.e. sample/line).

        Parameters
        ----------
        lat: float
             Latitude to be transformed.
        lon : float
              Longitude to be transformed.
        Returns
        -------
        x, y : tuple
               (Sample, line) position corresponding to the given (latitude, longitude).
        
        """
        geotransform = self.geotransform
        upperlat, upperlon, _ = self.inverse_coordinate_transformation.TransformPoint(lon, lat)
        x = (upperlat - geotransform[0]) / geotransform[1]
        y = (upperlon - geotransform[3]) / geotransform[5]
        return x, y

    def read_array(self, band=1, pixels=None, dtype='float32'):
        """
        Extract the required data as a numpy array

        Parameters
        ----------
        band : int
               The image band number to be extracted as a numpy array. Default band=1.

        pixels : list
                 [start, ystart, xstop, ystop]. Default pixels=None.

        dtype : str
                The numpy dtype for the output array. Default dtype='float32'.

        Returns
        -------
        array : NumPy array
                The dataset for the specified band.

        """
        band = self.dataset.GetRasterBand(band)

        dtype = getattr(numpy, dtype)

        if pixels == None:
            array = band.ReadAsArray().astype(dtype)
        else:
            xstart = pixels[0][0]
            ystart = pixels[0][1]
            xextent = pixels[1][0] - xstart
            yextent = pixels[1][1] - ystart
            array = band.ReadAsArray(xstart, ystart,
                                          xextent, yextent).astype(dtype)
        return array

def array_to_raster(array, file_name, projection=None,
                    geotransform=None, outformat='GTiff',
                    ndv=None):
    """

    Parameters
    ----------
    array : 

    file_name : str 

    projection : 
                 Default projection=None.

    geotransform : object 
                   Default geotransform=None.

    outformat : const char *
                Default outformat='GTiff'.

    ndv : float
          The no data value for the given band. See no_data_value(). Default ndv=None.

    """
    driver = gdal.GetDriverByName(outformat)
    try:
        y, x, bands = array.shape
        single = False
    except:
        bands = 1
        y, x = array.shape
        single = True

    #This is a crappy hard code to 32bit.
    dataset = driver.Create(file_name, x, y, bands, gdal.GDT_Float64)

    if geotransform:
        dataset.SetGeoTransform(geotransform)

    if projection:
        if isinstance(projection, str):
            dataset.SetProjection(projection)
        else:
            dataset.SetProjection(projection.ExportToWkt())

    if single == True:
        bnd = dataset.GetRasterBand(1)
        if ndv != None:
            bnd.SetNoDataValue(ndv)
        bnd.WriteArray(array)
        dataset.FlushCache()
    else:
        for i in range(1, bands + 1):
            bnd = dataset.GetRasterBand(i)
            if ndv != None:
                bnd.SetNoDataValue(ndv)
            bnd.WriteArray(array[:,:,i - 1])
            dataset.FlushCache()
