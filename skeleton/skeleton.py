import typing as tp
import numpy as np
import copy

ExtrasT = tp.Dict[str, tp.Any]
IdxFilterT = tp.Sequence[int]
LinksMapT = tp.Sequence[tp.Sequence[int]]
FloatArrT = np.ndarray[int, np.dtype[np.float64]]


class Joint():
    def __init__(
        self,
        name: str,
        position: tp.Optional[FloatArrT] = None,
        **kwargs,
    ) -> None:
        self.name = name

        self.position: FloatArrT = position or np.zeros(3)

        self.extras: ExtrasT = {}
        if kwargs:
            self.extras = copy.deepcopy(kwargs)

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def get_extras(
        self,
        key: str,
        default: tp.Optional[tp.Any] = None,
    ) -> tp.Any:
        return self.extras.get(key, default)


class Link():
    def __init__(
        self,
        top_joint: Joint,
        bottom_joint: Joint,
        name: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        self.top_joint = top_joint
        self.bottom_joint = bottom_joint

        self.name = name or self._create_name()

        self.extras: ExtrasT = {}
        if kwargs:
            self.extras = copy.deepcopy(kwargs)

    def _create_name(self) -> str:
        return f'{self.top_joint.name} -> {self.bottom_joint.name}'

    @property
    def vector(self) -> FloatArrT:
        return self.bottom_joint.position - self.top_joint.position

    @property
    def length(self) -> float:
        return tp.cast(float, np.linalg.norm(self.vector, 2))

    @property
    def cosines(self) -> FloatArrT:
        return self.vector / self.length

    @property
    def joints(self) -> tp.Tuple[Joint, Joint]:
        return self.top_joint, self.bottom_joint

    def get_extras(
        self,
        key: str,
        default: tp.Optional[tp.Any] = None,
    ) -> tp.Any:
        return self.extras.get(key, default)


class Skeleton():
    def __init__(
        self,
        name: str,
        joints: tp.Optional[tp.List[Joint]] = None,
        links: tp.Optional[tp.List[Link]] = None,
        index_map: tp.Optional[LinksMapT] = None,
    ) -> None:
        self.name = name

        self.joints: tp.List[Joint] = []
        self.links: tp.List[Link] = []

        if joints is not None:
            self.joints = joints

        if links is not None:
            self.links = links
            if joints is None:
                self.joints = self.joints_from_links(links)
        elif index_map is not None and joints is not None:
            self.links = self.links_from_indexes(index_map)

    @classmethod
    def from_links(cls, name: str, links: tp.List[Link]) -> 'Skeleton':
        joints = cls.joints_from_links(links)
        return cls(name, joints, links)

    @staticmethod
    def joints_from_links(links: tp.List[Link]) -> tp.List[Joint]:
        joints: tp.List[Joint] = []
        joints_used = set(joints)
        for link in links:
            for joint in link.joints:
                if joint not in joints_used:
                    joints.append(joint)
                joints_used.add(joint)
        return joints

    def _validate_indexes(self, idx1: int, idx2: int, max_idx: int) -> bool:
        return 0 <= idx1 < max_idx and 0 <= idx2 < max_idx

    def links_from_indexes(
        self,
        index_map: LinksMapT,
    ) -> tp.List[Link]:
        joints_num = len(self.joints)
        links: tp.List[Link] = []
        for top_idx, bottom_idx in index_map:
            if not self._validate_indexes(top_idx, bottom_idx, joints_num):
                raise IndexError(f'Index {top_idx, bottom_idx} out of Joints')
            links.append(Link(
                    self.joints[top_idx],
                    self.joints[bottom_idx],
            ))
        return links

    @property
    def joints_list(self) -> tp.List[str]:
        return [joint.name for joint in self.joints]

    @property
    def links_list(self) -> tp.List[str]:
        return [link.name for link in self.links]

    def get_joints_extras(
        self,
        key: str,
        default: tp.Optional[tp.Any] = None,
    ) -> tp.List[tp.Any]:
        return [joint.get_extras(key, default) for joint in self.joints]

    def get_links_extras(
        self,
        key: str,
        default: tp.Optional[tp.Any] = None,
    ) -> tp.List[tp.Any]:
        return [link.get_extras(key, default) for link in self.links]

    def update_joints_extras(
        self,
        extras: tp.Sequence[ExtrasT],
        idx_map: tp.Optional[IdxFilterT] = None,
        with_replacement: bool = False,
    ) -> None:
        if idx_map is None:
            if len(self.joints) != len(extras):
                raise RuntimeError
            for joint, extra in zip(self.joints, extras):
                if not with_replacement:
                    joint.extras.update(extra)
                else:
                    joint.extras = copy.deepcopy(extra)
            return
        for idx, extra in zip(idx_map, extras):
            if not with_replacement:
                self.joints[idx].extras.update(extra)
            else:
                self.joints[idx].extras = copy.deepcopy(extra)

    def update_links_extras(
        self,
        extras: tp.Sequence[ExtrasT],
        idx_map: tp.Optional[IdxFilterT] = None,
        with_replacement: bool = False,
    ) -> None:
        if idx_map is None:
            if len(self.links) != len(extras):
                raise RuntimeError
            for link, extra in zip(self.links, extras):
                if not with_replacement:
                    link.extras.update(extra)
                else:
                    link.extras = copy.deepcopy(extra)
            return
        for idx, extra in zip(idx_map, extras):
            if not with_replacement:
                self.links[idx].extras.update(extra)
            else:
                self.links[idx].extras = copy.deepcopy(extra)

    def update_positions(
        self,
        positions: tp.Sequence[FloatArrT],
        idx_map: tp.Optional[IdxFilterT] = None,
    ) -> None:
        if idx_map is None:
            if len(self.joints) != len(positions):
                raise RuntimeError
            for joint, position in zip(self.joints, positions):
                joint.position = np.array(position)
            return
        for idx, position in zip(idx_map, positions):
            self.joints[idx].position = np.array(position)

    def get_centroid(
        self,
        idx_map: tp.Optional[IdxFilterT] = None,
    ) -> FloatArrT:
        if idx_map is None:
            return np.average([joint.position for joint in self.joints], axis=0)
        return np.average([self.joints[idx].position for idx in idx_map], axis=0)

    def get_length(
        self,
        idx_map: tp.Optional[IdxFilterT] = None,
    ) -> FloatArrT:
        if idx_map is None:
            return np.array([link.length for link in self.links])
        return np.array([self.links[idx].length for idx in idx_map])

    def get_cosines(
        self,
        idx_map: tp.Optional[IdxFilterT] = None,
    ) -> FloatArrT:
        if idx_map is None:
            return np.array([link.cosines for link in self.links])
        return np.array([self.links[idx].cosines for idx in idx_map])
