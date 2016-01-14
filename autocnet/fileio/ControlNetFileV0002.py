from protobuf3.fields import MessageField, EnumField, StringField, BoolField, Int32Field, DoubleField
from enum import Enum
from protobuf3.message import Message


class ControlNetFileHeaderV0002(Message):
    pass


class ControlPointFileEntryV0002(Message):

    class PointType(Enum):
        Free = 2
        Constrained = 3
        Fixed = 4
        obsolete_Tie = 0
        obsolete_Ground = 1

    class AprioriSource(Enum):
        NA = 0
        User = 1
        AverageOfMeasures = 2
        Reference = 3
        Ellipsoid = 4
        DEM = 5
        Basemap = 6
        BundleSolution = 7

    class PointLogData(Message):
        pass

    class Measure(Message):

        class MeasureType(Enum):
            Candidate = 0
            Manual = 1
            RegisteredPixel = 2
            RegisteredSubPixel = 3

        class MeasureLogData(Message):
            pass

ControlNetFileHeaderV0002.add_field('networkId', StringField(field_number=1, required=True))
ControlNetFileHeaderV0002.add_field('targetName', StringField(field_number=2, required=True))
ControlNetFileHeaderV0002.add_field('created', StringField(field_number=3, optional=True))
ControlNetFileHeaderV0002.add_field('lastModified', StringField(field_number=4, optional=True))
ControlNetFileHeaderV0002.add_field('description', StringField(field_number=5, optional=True))
ControlNetFileHeaderV0002.add_field('userName', StringField(field_number=6, optional=True))
ControlNetFileHeaderV0002.add_field('pointMessageSizes', Int32Field(field_number=7, repeated=True))

ControlPointFileEntryV0002.PointLogData.add_field('doubleDataType', Int32Field(field_number=1, optional=True))
ControlPointFileEntryV0002.PointLogData.add_field('doubleDataValue', DoubleField(field_number=2, optional=True))
ControlPointFileEntryV0002.PointLogData.add_field('boolDataType', Int32Field(field_number=3, optional=True))
ControlPointFileEntryV0002.PointLogData.add_field('boolDataValue', BoolField(field_number=4, optional=True))
ControlPointFileEntryV0002.Measure.MeasureLogData.add_field('doubleDataType', Int32Field(field_number=1, optional=True))
ControlPointFileEntryV0002.Measure.MeasureLogData.add_field('doubleDataValue', DoubleField(field_number=2, optional=True))
ControlPointFileEntryV0002.Measure.MeasureLogData.add_field('boolDataType', Int32Field(field_number=3, optional=True))
ControlPointFileEntryV0002.Measure.MeasureLogData.add_field('boolDataValue', BoolField(field_number=4, optional=True))
ControlPointFileEntryV0002.Measure.add_field('serialnumber', StringField(field_number=1, required=True))
ControlPointFileEntryV0002.Measure.add_field('type', EnumField(field_number=2, required=True, enum_cls=ControlPointFileEntryV0002.Measure.MeasureType))
ControlPointFileEntryV0002.Measure.add_field('sample', DoubleField(field_number=3, optional=True))
ControlPointFileEntryV0002.Measure.add_field('line', DoubleField(field_number=4, optional=True))
ControlPointFileEntryV0002.Measure.add_field('sampleResidual', DoubleField(field_number=5, optional=True))
ControlPointFileEntryV0002.Measure.add_field('lineResidual', DoubleField(field_number=6, optional=True))
ControlPointFileEntryV0002.Measure.add_field('choosername', StringField(field_number=7, optional=True))
ControlPointFileEntryV0002.Measure.add_field('datetime', StringField(field_number=8, optional=True))
ControlPointFileEntryV0002.Measure.add_field('editLock', BoolField(field_number=9, optional=True))
ControlPointFileEntryV0002.Measure.add_field('ignore', BoolField(field_number=10, optional=True))
ControlPointFileEntryV0002.Measure.add_field('jigsawRejected', BoolField(field_number=11, optional=True))
ControlPointFileEntryV0002.Measure.add_field('diameter', DoubleField(field_number=12, optional=True))
ControlPointFileEntryV0002.Measure.add_field('apriorisample', DoubleField(field_number=13, optional=True))
ControlPointFileEntryV0002.Measure.add_field('aprioriline', DoubleField(field_number=14, optional=True))
ControlPointFileEntryV0002.Measure.add_field('samplesigma', DoubleField(field_number=15, optional=True))
ControlPointFileEntryV0002.Measure.add_field('linesigma', DoubleField(field_number=16, optional=True))
ControlPointFileEntryV0002.Measure.add_field('log', MessageField(field_number=17, repeated=True, message_cls=ControlPointFileEntryV0002.Measure.MeasureLogData))
ControlPointFileEntryV0002.add_field('id', StringField(field_number=1, required=True))
ControlPointFileEntryV0002.add_field('type', EnumField(field_number=2, required=True, enum_cls=ControlPointFileEntryV0002.PointType))
ControlPointFileEntryV0002.add_field('chooserName', StringField(field_number=3, optional=True))
ControlPointFileEntryV0002.add_field('datetime', StringField(field_number=4, optional=True))
ControlPointFileEntryV0002.add_field('editLock', BoolField(field_number=5, optional=True))
ControlPointFileEntryV0002.add_field('ignore', BoolField(field_number=6, optional=True))
ControlPointFileEntryV0002.add_field('jigsawRejected', BoolField(field_number=7, optional=True))
ControlPointFileEntryV0002.add_field('referenceIndex', Int32Field(field_number=8, optional=True))
ControlPointFileEntryV0002.add_field('aprioriSurfPointSource', EnumField(field_number=9, optional=True, enum_cls=ControlPointFileEntryV0002.AprioriSource))
ControlPointFileEntryV0002.add_field('aprioriSurfPointSourceFile', StringField(field_number=10, optional=True))
ControlPointFileEntryV0002.add_field('aprioriRadiusSource', EnumField(field_number=11, optional=True, enum_cls=ControlPointFileEntryV0002.AprioriSource))
ControlPointFileEntryV0002.add_field('aprioriRadiusSourceFile', StringField(field_number=12, optional=True))
ControlPointFileEntryV0002.add_field('latitudeConstrained', BoolField(field_number=13, optional=True))
ControlPointFileEntryV0002.add_field('longitudeConstrained', BoolField(field_number=14, optional=True))
ControlPointFileEntryV0002.add_field('radiusConstrained', BoolField(field_number=15, optional=True))
ControlPointFileEntryV0002.add_field('aprioriX', DoubleField(field_number=16, optional=True))
ControlPointFileEntryV0002.add_field('aprioriY', DoubleField(field_number=17, optional=True))
ControlPointFileEntryV0002.add_field('aprioriZ', DoubleField(field_number=18, optional=True))
ControlPointFileEntryV0002.add_field('aprioriCovar', DoubleField(field_number=19, repeated=True))
ControlPointFileEntryV0002.add_field('adjustedX', DoubleField(field_number=20, optional=True))
ControlPointFileEntryV0002.add_field('adjustedY', DoubleField(field_number=21, optional=True))
ControlPointFileEntryV0002.add_field('adjustedZ', DoubleField(field_number=22, optional=True))
ControlPointFileEntryV0002.add_field('adjustedCovar', DoubleField(field_number=23, repeated=True))
ControlPointFileEntryV0002.add_field('log', MessageField(field_number=24, repeated=True, message_cls=ControlPointFileEntryV0002.PointLogData))
ControlPointFileEntryV0002.add_field('measures', MessageField(field_number=25, repeated=True, message_cls=ControlPointFileEntryV0002.Measure))
