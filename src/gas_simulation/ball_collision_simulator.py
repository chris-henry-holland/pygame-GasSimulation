#!/usr/bin/env python3

from typing import (
    Union,
    Tuple,
    List,
    Set,
)

import heapq
import itertools
import math

from gas_simulation.utils import Real

def closestApproachVector(r1: Tuple[Real], r2: Tuple[Real],\
        v: Tuple[Real], v_abs_sq: Real)\
        -> Tuple[Union[Tuple[Real], Real]]:
    """
    Given two points in the n-dimensional real vector space (i.e.
    R ** n where n is a strictly positive integer) at position vectors
    r1 and r2, and a vector v in the same vector space, finds the real
    number a such that there are no values of x such that the vector:
        r2 - r1 + x * v
    has a magnitude smaller than that for which x = a. Returns this
    vector and the real numeric value of a.
    
    Each vector is expressed in terms of the same orthonormal basis,
    with every position vector being expressed relative to
    the same origin.
    
    Args:
    	Required positional:
    	r1 (n-tuple of real numeric values): Position vector r1
    	        expressed in terms of the orthonormal basis.
        r2 (n-tuple of real numeric values): Position vector r2
                expressed in terms of the orthonormal basis.
        v (n-tuple of real numeric values): Vector v expressed in
                terms of the orthonormal basis.
        v_abs_sq (real numeric value): The squared magnitude of the
                vector v.
    
    Returns:
    2-tuple whose index 0 contains the vector (r2 - r1 + a * v) for
    the calculated value of a in terms of the orthonormal basis and
    whose index 1 contains that calculated real numeric value a.
    """
    displ = sum(x * (y - z) for x, y, z in zip(v, r2, r1)) # dot product
    dt = displ / v_abs_sq
    r1_mod = [x + y * dt for x, y in zip(r1, v)]
    return tuple(x - y for x, y in zip(r2, r1_mod)), dt

class Ball:
    """
    Object representing a ball in an instance of MultiBallSimulation,
    a simulation in real n-dimensional space of perfectly rigid
    n-dimensional balls with mass in a perfectly rigid, infinitely
    massive n-hyperrectangular box (where n is a strictly positive
    integer) in a uniform gravitational field according to Newtonian
    mechanics, where every collision (both between balls and between
    balls and walls) is perfectly elastic and frictionless and there
    are no non-contact forces between any pair of balls or between any
    ball and the box. Given this model, rotational motion of any
    ball in the simulation is constant throughout and does not
    influence translational motion of itself or any other object, and
    is therefore ignored (with rotational kinetic energy not included
    in the kinetic or mechanical energy calculations).
    
    For full details of the simulation itself, see the documentation of
    the MultiBallSimulation class.
    
    A ball has a centre at a given point in space (represented by
    a position vector, stored at the attribute r) and a radius (a
    strictly positive real numeric value, stored at attribute radius)
    and is defined to be the set of all points in the n-dimensional
    space whose distance from the position of the centre at that time
    is strictly less than the radius. The velocity of a ball at
    a given time refers to the velocity of its centre (i.e. the
    derivative of the position vector representing the centre with
    respect to time), with the acceleration of the ball similarly
    defined.
    
    The set of points comprising the ball should be completely within
    the walls of the box and should not intersect with the
    corresponding set of position vectors comprising any
    other objects (with such an intersection referred to as an
    overlap), with motion that would give rise to a state in
    conflict with these requirements being resolved by an elastic
    collision at the latest time before the requirements would not
    be satisfied (i.e. an elastic collsion with the wall when the
    ball is exactly radius away from the wall or an elastic collision
    with the other ball at the point where their separation is exactly
    the sum of their respective radii).
    
    An elastic collision is one in which the total momentum and kinetic
    energy immediately after the collision are the same as before the
    collision (recalling that the box is assumed to have infinite
    mass). Furthermore, the collisions are assumed to be instantaneous
    and with a change in the velocity vector which in the case of
    collision with a wall, is a vector normal to the wall at that point
    and in the case of a collision with another ball is a non-negative
    real scalar multiple of the vector from the centre of the ball
    being collided with to the centre of the ball in question at the
    time of the collision.
    
    In the absence of collisions, the motion of a ball is that of
    a rigid body under constant acceleration by a gravitational field
    as specified by g. Outside of collisions, there is no force between
    balls or between any ball and the walls of the box.
    
    Mass, distance, and time are measured in terms of the same units
    as the simulation (see the documentation of the MultiBallSimulation
    class), with other units being derived from these, for example a
    speed/velocity unit being a unit of distance divided by a unit of
    time an acceleration unit being a unit of distance divided by a
    squared unit of time, a momentum unit being a unit of mass times a
    unit of velocity and an energy unit being a unit of mass being a
    unit of mass times a unit of velocity squared. We refer to these as
    the simulation's units.
    
    All vectors relating to the n-dimensional space (e.g. position
    or velocity vectors) are defined in terms of the orthonormal basis
    of the simulation (from which the labelling of basis vectors with
    integers 0 to (n - 1) inclusive is also carried over), with
    position vectors given relative to the spatial origin of the
    simulation, and the time measure is the same as that of the
    simulation (see the documentation of the MultiBallSimulation
    class for these definitions). The vectors relating to the
    n-dimensional space are represented by n-tuples of real numeric
    values, for which each item represents the signed numerical
    value of the projection of the vector onto the corresponding basis
    vector in terms of the corresponding units of the simulation.
    
    Initialisation args:
    
        Required positional:
        
        sim (MultiBallSimulation): Sets the attribute sim, representing
                the simulation of which the Ball object is a part, and
                from which the values for the dimensions of the box and
                the uniform gravitational field are derived.
        m (strictly positive real numeric value): Sets the attribute m
                representing the mass of the ball in terms of the
                simulation's mass units.
        radius (strictly positive real numeric value): Sets the 
                attribute radius, representing the radius of the ball
                in terms of the simulation's distance units.
        r0 (n-tuple of real numeric values): The position vector (i.e.
                the location relative to the origin in terms of the
                basis vectors) of the centre of the ball at time t0
                in terms of the simulation's distance units.
        v0 (n-tuple of real numeric values): The velocity vector of the
                centre of the ball at time t0 in terms of the
                simulation's velocity units.
        
        Optional named:
        
        t0 (real numeric value): The time at which the simulation of
                the ball starts, when time the position and velicity
                vectors of the centre of the ball are r0 and v0
                respectively.
            Default: 0
    
    Attributes:
    
        Fixed (i.e. set at instance creation and unchanged through the
            lifetime of the object instance):
            
        sim (MultiBallSimulation): the simulation of which the Ball
                object is a part, and from which the values for the
                dimensions of the box and the uniform gravitational
                field are derived.
        n_dims (strictly positive int): The number of dimensions of
                the simulation (equal to the corresponding attribute
                of the attribute sim).
        m (strictly positive real numeric value): The mass of the ball
                in terms of the simulation's mass units units.
        radius (strictly positive real numeric value): The radius
                of the ball in terms of the simultaion's distance
                units.
        g (n-tuple of real numeric values): Vector representing the
                uniform gravitational field in terms of the
                simulation's acceleration units.
        centre_ranges (n-tuple of 2-tuples of real numeric values):
                The closed ranges such that the ball is entirely inside
                the box if and only if every one the components of the
                position vector of its centre (attribute r) is within
                the corresponding closed range (allowing for the radius
                of the ball).
        
        Updateable (i.e. can be updated by the user at any point
            during the lifetime of the instance):
            
        t (real numeric value): The current time for the ball in terms
                of the simulation's time measure.
        
        Derived (i.e. set at instance creation and updated based on
            calculations made using other attribute values):
            
        r (n-tuple of real numeric values): The position vector
                (i.e. the location relative to the origin in terms of
                the basis vectors) of the centre of the ball at time
                corresponding to attribute t in terms of the
                simulation's distance units.
        v (n-tuple of real numeric values): The velocity vector
                of the centre of the ball at time corresponding to
                attribute t in terms of the simulation's velocity
                units.
        next_wall_heap (list of 2-tuples containing a real numeric
                value (index 0) and int (index1), arranged as a
                min-heap by the heapq package): Min-heap whose entries
                are 2-tuples whose index 1 contains the index of a
                basis vector and whose index 0 contains the time (in
                terms of the simulation's time measure) of the next
                collision with one of the walls normal to that basis
                vector. The min-heap is arranged based on the time
                (with the element at index 0 of the list, assuming
                the list has any elements, containing a time no
                greater than any other element of the list).
                Each basis vector index (i.e. every integer between 0
                and one less than the number of dimensions of the
                simulation) is represented at most once, with every
                index not represented being due to the current
                trajectory of the ball never colliding with either
                of the walls to which that basis vector is normal
                (which can only occur if the component of both the
                velocity and acceleration vectors of the Ball object
                along that basis vector are both exactly zero).
    
    Methods:
        (For full description, see documentation of the method itself)
        
        initialiseNextWallHeap(): Resets the attribute next_wall_heap.
        updateNextWallHeapSingleDimension(): Calculates the next time
                (if any) after the current time that the ball would
                collide with either of the walls normal to a user
                specified basis vector assuming the Ball object does
                not collide with any other objects before that time.
                If such a collision exists, adds the details of this
                wall collision to the attribute next_wall_heap.
        positionAtTime(): Finds the position vector of the Ball object
                at a given time assuming the Ball object does not
                collide with any walls or other objects between the
                time equal to attribute t and the time specified (i.e.
                follows its current trajectory).
        velocityAtTime(): Finds the velocity vector of the Ball object
                at a given time assuming the Ball object does not
                collide with any walls or other objects between the
                time equal to attribute t and the time specified (i.e.
                follows its current trajectory).
        positionAndVelocityAtTime(): Finds the position and velocity
                vectors of the Ball object at a given time assuming the
                Ball object does not collide with any walls or other
                objects between the time equal to attribute t and the
                time specified (i.e. follows its current trajectory).
        progressToNextWallCollision(): Increases current time to
                correspond to the time immediately after next occasion
                the Ball object collides with a wall, updating the
                time, position and velocity having taken into account
                the trajectory of the object before colliding with the
                wall and the elastic collision of the Ball object with
                the wall.
        identifyOtherBallNextCollision(): Given another Ball object,
                calculates if and when the next collision between the
                two will occur, assuming neither collide with any other
                objects or walls before that time.
        progressToNextOtherBallCollision(): Increases current time to
                correspond to the time immediately after next occasion
                the Ball object another specific Ball object, updating
                the time, position and velocity of both Ball objects
                having taken into account the trajectories of both
                objects before colliding with each other and the
                elastic collision of two objects with each other.
        isOutsideBox(): Checks whether any part of the object is
                outside the n-hyperrectangular box, which can act as a
                way of detecting that collision detection and/or
                resolution is not functioning correctly.
        calculateKineticEnergy(): Calculates the translational kinetic
                energy of the Ball object at the time equal to
                attribute t (the current time).
        calculatePotentialEnergy(): Calculates the gravitational
                potential energy of the Ball object at the time equal
                to attribute t (the current time).
        calculateMechanicalEnergy(): Calculates the mechanical energy
                (translational kinetic plus potential energy) of the
                Ball object at the time equal to attribute t (the
                current time).
    """
    def __init__(self, sim: "MultiBallSimulation", m: Real,\
            radius: Real, r0: Tuple[Real],\
            v0: Tuple[Real], t0: Real=0):
        self._sim = sim
        self._m = m
        self._radius = radius
        
        # The attribute _t0 is the current reference time for the ball,
        # at which time the position and velocity vectors are stored as
        # the attributes _r0 and _v0 respectively. From these values
        # the current trajectory is calculated and the next collision
        # is identified.
        # These are updated at only at the start of the simulation and
        # immediately after every collision the ball experiences.
        # The attributes r and v (representing the position and
        # velocity vectors at the current time t) are calculated based
        # on the trajectory of the ball from time equal to the
        # attribute _t0 to the time equal to attribute t, under the
        # assumption that the ball does not experience any collisions
        # between times of attributes _t0 and t (i.e. motion in a
        # uniform gravitational field).
        self._t0 = t0
        self._r0 = r0
        self._v0 = v0
        
        self.t = self._t0
        
        self.next_wall_heap = []
        self.initialiseNextWallHeap()
    
    @property
    def sim(self):
        return self._sim
    
    @property
    def m(self):
        return self._m
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def n_dims(self):
        return self.sim.n_dims
    
    @property
    def centre_ranges(self):
        res = getattr(self, "_centre_ranges", None)
        if res is None:
            box_ranges = self.sim.box_interior_ranges
            rad = self.radius
            res = tuple((rng[0] + rad, rng[1] - rad)\
                    for rng in box_ranges)
            self._centre_ranges = res
        return res
    
    @property
    def g(self):
        return self.sim.g
    
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        if t == getattr(self, "_t", None): return
        self._t = t
        self._r = None
        self._v = None
    
    @property
    def r(self):
        res = getattr(self, "_r", None)
        if res is None:
            res = self.positionAtTime(self.t)
            self._r = res
        return res
    
    @property
    def v(self):
        res = getattr(self, "_v", None)
        if res is None:
            res = self.velocityAtTime(self.t)
            self._v = res
        return res
    
    @property
    def _t_ref(self):
        return getattr(self.sim, "_t_ref", float("inf"))
    
    def _timeToNextWall(self, idx: int) -> Real:
        """
        Method identifying the amount of time after the current
        reference time, attribute _t0, at which the Ball object
        will collide with one of the walls normal to the idx:th
        basis vector (if any), assuming the ball does not collide
        with any other objects (not including the other walls) during
        this time interval.
        
        Args:
            Required positional:
            idx (int): Integer between 0 and (self.n_dims - 1)
                    inclusive specifying the index of the basis vector
                    to which the walls considered are to be normal.
        
        Returns:
        Non-negative real numeric value (int or float) giving the time
        interval after the reference time (attribute _t0) after which
        a collision between the Ball object and one of the specified
        walls will next occur (assuming the Ball object experiences no
        collisions with other object during this time interval) in
        terms of the units used (in this case time units).
        If no such collision will occur (due to the velocity and
        acceleration in that direction both being exactly zero) then
        float("inf") is returned to represent this.
        """
        v = self._v0[idx]
        r = self._r0[idx]
        a = self.g[idx]
        neg = (v < 0)
        if not a:
            wall = self.centre_ranges[idx][not neg]# + 1
            if not v: return float("inf")
            return (wall - r) / v
        vg_align = ((a > 0) == (v >= 0))
        for j in range(2):
            wall = self.centre_ranges[idx][not (neg ^ j)]# + 1
            d = wall - r
            discr = v ** 2 + 2 * a * d
            if discr < 0: continue
            s = math.sqrt(discr)
            res_lst = sorted([(s - v) / a, (-v - s) / a])
            res = res_lst[1] if vg_align or j else res_lst[0]
            break
        return res
    
    def initialiseNextWallHeap(self) -> None:
        """
        Completely resets the attribute next_wall_heap and sets up its
        min-heap structure, based on the current time, position and
        velocity of the Ball object.
        
        Returns:
        None
        """
        heap = []
        for i in range(self.n_dims):
            if not self._v0[i] and not self.g[i]: continue
            neg = self._v0[i] < 0
            wall = self.centre_ranges[i][not neg]
            heap.append((self._t0 + self._timeToNextWall(i), i))
        heapq.heapify(heap)
        self.next_wall_heap = heap
        return
    
    def updateNextWallHeapSingleDimension(self, idx: int):
        """
        Inserts the details of the next collision with one of the walls
        normal to the idx:th basis vector (if any) to the attribute
        next_wall_heap (maintaining the attribute's min-heap
        structure), assuming the Ball object does not collide with any
        other objects (not including other walls) before the time of
        that collision.        
        
        Args:
            Required positional:
            idx (int): Integer between 0 and (self.n_dims - 1)
                    inclusive specifying the index of the basis vector
                    to which the walls considered are to be normal.
        
        Returns:
        None
        """
        if not self._v0[idx] and not self.g[idx]:
            return
        heapq.heappush(self.next_wall_heap,\
                (self._t0 + self._timeToNextWall(idx), idx))
        return
    
    def _positionAfterTimeIncrement(self, dt: Real) -> Tuple[Real]:
        """
        Calculates the position vector of the Ball object after a time
        interval dt from the reference time (represented by attribute
        _t0) assuming that the Ball object does not experience any
        collisions with walls or any other objects during this time
        interval.
        
        Args:
            Required positional:
            dt (real numeric value): The time interval in the
                    simulation's time units.
        
        Returns:
        n-tuple (where n is the number of spatial dimensions of the
        simulation, attribute n_dims) of real numeric values
        representing the position vector of the Ball object after a
        time interval dt from the time represented by attribute _t0
        assuming that the Ball object does not experience any
        collisions with walls or any other objects during this time
        interval.
        """
        return tuple(r + v * dt + a * dt ** 2 / 2 for r, v, a in\
                zip(self._r0, self._v0, self.g))
    
    def _velocityAfterTimeIncrement(self, dt: Real) -> Tuple[Real]:
        """
        Calculates the velocity vector of the Ball object after a time
        interval dt from the reference time (represented by attribute
        _t0) assuming that the Ball object does not experience any
        collisions with walls or any other objects during this time
        interval.
        
        Args:
            Required positional:
            dt (real numeric value): The time interval in the
                    simulation's time units.
        
        Returns:
        n-tuple (where n is the number of spatial dimensions of the
        simulation, attribute n_dims) of real numeric values
        representing the velocity vector of the Ball object after a
        time interval dt from the time represented by attribute _t0
        assuming that the Ball object does not experience any
        collisions with walls or any other objects during this time
        interval.
        """
        return tuple(v + a * dt for v, a in zip(self._v0, self.g))
    
    def _positionAndVelocityAfterTimeIncrement(self, dt: Real)\
            -> Tuple[Tuple[Real]]:
        """
        Calculates the position and velocity vectors of the Ball object
        after a time interval dt from the reference time (represented
        by attribute _t0) assuming that the Ball object does not
        experience any collisions with walls or any other objects
        during this time interval.
        
        Args:
            Required positional:
            dt (real numeric value): The time interval in the
                    simulation's time units.
        
        Returns:
        2-tuple of n-tuples (where n is the number of spatial
        dimensions of the simulation, attribute n_dims) of real numeric
        values representing the position (index 0) and velocity
        (index 1) vectors of the Ball object after a time interval dt
        from the time corresponding to attribute _t0 assuming that the
        Ball object does not experience any collisions with walls or
        any other objects during this time interval.
        """
        return (self._positionAfterTimeIncrement(dt),\
                self._velocityAfterTimeIncrement(dt))
    
    def positionAtTime(self, t: Real) -> Tuple[Real]:
        """
        Calculates the position vector of the Ball object at time t
        assuming that the Ball object does not experience any
        collisions with walls or any other objects between the current
        time and time t.
        
        Args:
            Required positional:
            t (real numeric value): The time in terms of the
                    simulation's time measure for which the position
                    vector is to be calculated.
        
        Returns:
        n-tuple (where n is the number of spatial dimensions of the
        simulation, attribute n_dims) of real numeric values
        representing the position vector of the Ball object at the
        time represented by t.
        """
        if t == self.t and getattr(self, "_r", None) is not None:
            return self.r
        elif t == self._t0: return self._r0
        dt = t - self._t0
        return self._positionAfterTimeIncrement(dt)
    
    def velocityAtTime(self, t: Real) -> Tuple[Real]:
        """
        Calculates the velocity vector of the Ball object at time t
        assuming that the Ball object does not experience any
        collisions with walls or any other objects between the current
        time and time t.
        
        Args:
            Required positional:
            t (real numeric value): The time in terms of the
                    simulation's time measure for which the velocity
                    vector is to be calculated.
        
        Returns:
        n-tuple (where n is the number of spatial dimensions of the
        simulation, attribute n_dims) of real numeric values
        representing the velocity vector of the Ball object at the
        time represented by t.
        """
        if t == self.t and getattr(self, "_v", None) is not None:
            return self.v
        elif t == self._t0: return self._v0
        dt = t - self._t0
        return self._velocityAfterTimeIncrement(dt)
    
    def positionAndVelocityAtTime(self, t: Real)\
            -> Tuple[Tuple[Real]]:
        """
        Calculates the position and velocity vectors of the Ball object
        at time t assuming that the Ball object does not experience any
        collisions with walls or any other objects between the current
        time and time t.
        
        Args:
            Required positional:
            t (real numeric value): The time in terms of the
                    simulation's time measure for which the position
                    and velocity vectors are to be calculated.
        
        Returns:
        2-tuple of n-tuples (where n is the number of spatial
        dimensions of the simulation, attribute n_dims) of real numeric
        values representing the position (index 0) and velocity
        (index 1) vectors of the Ball object at the time represented by
        t.
        """
        if t == self.t and getattr(self, "_r", None) is not None:
            return self.r, self.v
        elif t == self._t0: return self._r0, self._v0
        dt = t - self._t0
        return self._positionAndVelocityAfterTimeIncrement(dt)
    
    def _updateTime0(self, t0: Real) -> None:
        """
        Resets the reference time to the time represented by t0, also
        updating the reference position and velocity. Assumes no
        collisions with walls or other objects occur between the
        current reference time and the new reference time (i.e. it
        follows its current trajectory).
        
        Args:
            Required positional:
            t0 (real numeric value): The new time to which the
                    reference time is to be set in terms of the
                    simulation's time measure.
        
        Returns:
        None
        """
        self._r0, self._v0 =\
                self._positionAndVelocityAfterTimeIncrement(\
                t0 - self._t0)
        self._t0 = t0
        self._r = None
        self._v = None
        return
    
    def progressToNextWallCollision(self) -> None:
        """
        Increases current time to correspond to the time immediately
        after next occasion the Ball object collides with a wall,
        updating the time, position and velocity attributes having
        taken into account the trajectory of the Ball object up until
        the collision with the wall and the effect of the elastic
        collision of the Ball object with the wall on its velocity.
        This assumes the Ball object does not collide with any other
        objects in the intervening time.
        The accurate details of this collision should have been
        previously prepared using by defining and keeping updated the
        attribute next_wall_heap throught the appropriate use of the
        initialiseNextWallHeap() and
        updateNextWallHeapSingleDimension().
        In general simulations, this should be used only when collision
        between this Ball object and a wall is the next collision (both
        between pairs of objects and between objects and walls) due to
        occur in the simulation as a whole.
        
        Returns:
        None
        """
        if not self.next_wall_heap: return
        t, i = heapq.heappop(self.next_wall_heap)
        self._updateTime0(t)
        v = list(self._v0)
        v[i] = -v[i]
        self._v0 = tuple(v)
        self.updateNextWallHeapSingleDimension(i)
        return
    
    def identifyOtherBallNextCollision(self, other: "Ball")\
            -> Tuple[Union[Real, Tuple[Real]]]:
        """
        Given another Ball object calculates if and when the next
        collision between this Ball object and that Ball object will
        will occur before either object next collides with a wall,
        assuming neither collide with any other objects before that
        time. If such a collision is identified, returns this time
        and sufficient information for the outcome of the collision
        to be calculated.
        
        Note that this calculation is unaffected by the presence
        of a uniform gravitational field (and the value of this
        field is not used in the calculation). This is a direct
        consequence of the Equivalence Principle, which states
        that motion in an inertial frame within a uniform gravitational
        field is indistinguishable from motion with no gravitational
        field in a frame of reference accelerating at -g (where g is
        the gravitational field vector). As such, in the absence of
        collisions, the trajectories of two Ball objects relative
        to each other is the same as if there were no gravitational
        field (which, given that there are no other forces acting
        on either ball, means that relative velocity is constant).
        
        Args:
            Required positional:
            other (Ball): The other Ball object with which the
                    next collision with the current Ball is to be
                    identified.
        
        Returns:
        If the two Ball objects on their current trajectories will not
        collide at all or before either of them is due to next collide
        with a wall, an empty tuple.
        Otherwise, a 4-tuple whose index 0 contains a real numeric
        value representing the time at which the collision would occur
        (in terms of the simulation time measure), whose index 1
        contains a n-tuple of real numeric values (where n is the
        number of spatial dimensions of the simulation) representing
        the displacement vector from the centre of this Ball to the
        other Ball at the positions they are due to collide, and whose
        index 2 and 3 contain n-tuples of real numeric values
        representing the velocity vectors of this Ball object and the
        other Ball object respectively in the zero momentum frame of
        the two objects combined (i.e. the frame of reference moving
        in such a way that the overall momentum of both objects is
        exactly zero).
        """
        rad_sum = self.radius + other.radius
        rad_sum_sq = rad_sum ** 2
        t_max = self.next_wall_heap[0][0] if self.next_wall_heap\
                else float("inf")
        t_max = min(t_max, other.next_wall_heap[0][0]\
                if other.next_wall_heap else float("inf"))
        t0 = max(self._t0, other._t0)
        if t0 >= t_max: return ()
        r1, v1 = self.positionAndVelocityAtTime(t0)
        r2, v2 = other.positionAndVelocityAtTime(t0)
        
        # Find the zero momentum frame
        m_tot = self.m + other.m
        m_ratio = other.m / self.m
        v0 = tuple((self.m * x + other.m * y) / m_tot\
                for x, y in zip(v1, v2))
        
        #v2_zmf = tuple(x - y for x, y in zip(v2, v0))
        
        v_net = tuple(x - y for x, y in zip(v1, v2))
        
        # Check whether balls are separating along any dimension along
        # which they are already too far apart to collide
        for i in range(self.n_dims):
            if (r1[i] - r2[i] > rad_sum and v_net[i] >= 0) or\
                    r2[i] - r1[i] > rad_sum and v_net[i] <= 0:
                return ()
        # Check that balls are moving such that (at least initially)
        # their relative distance is decreasing
        if sum((x1 - x2) * y for x1, x2, y in zip(r1, r2, v_net)) >= 0:
            return ()
        v_net_sq = sum(x ** 2 for x in v_net)
        approach_vec, t2 =\
                closestApproachVector(r1, r2, v_net, v_net_sq)
        #if t2 < 0:
        #    return ()
        approach_vec_sq = sum(x ** 2 for x in approach_vec)
        # Check whether closest approach is less than the sum of the
        # radii of the two balls
        if approach_vec_sq >= rad_sum_sq:
            return ()
        dt2 = math.sqrt((rad_sum_sq - approach_vec_sq) / v_net_sq)
        t2 += t0 - dt2
        if not t2 < t_max:
            return ()
        # Displacement vector from the centre of ball self to the
        # centre of ball other at the point of collision
        contact_displ_vec = tuple(x + y * dt2 for x, y in\
                zip(approach_vec, v_net))
        v1_zmf = tuple(x - y for x, y in zip(v1, v0))
        return t2, contact_displ_vec, v1_zmf#, v2_zmf
    
    def progressToNextOtherBallCollision(self, other: "Ball",\
            t_collide: Real, contact_displ_vec: Tuple[Real],\
            v1_zmf: Real) -> None:
        """
        Increases current time to correspond to the time immediately
        after next occasion this Ball object collides with another
        specific Ball object, updating the time, position and velocity
        attributes of both objects having taken into account the
        trajectories of the two Ball objects up until the collision
        between them and the effect of the elastic collision with each
        other on their respective velocities.
        This assumes the Ball object does not collide with any other
        objects in the intervening time.
        The accurate details of this collision should have been
        previously calculated using the method
        identifyOtherBallNextCollision() for this Ball object using
        the other Ball object as its first argument, the output of
        which should be used for the arguments of this method following
        the first. Use of the method identifyOtherBallNextCollision()
        should have occurred at a time after the latest collision for
        both of the Ball objects.
        In general simulations, this should be used only when collision
        between these two Ball objects is the next collision (both
        between pairs of objects and between objects and walls) due to
        occur in the simulation as a whole.
        
        Args:
            Required positional:
            other (Ball object): The Ball object with which this Ball
                    object is to collide.
            t_collide (real numeric value): The previously calculated
                    time (in terms of the simulation's time measure) at
                    which the collision occurs.
            contact_displ_vec (n-tuple of real numeric types where n is
                    the number of spatial dimensions of the
                    simulation): The previously calculated displacement
                    vector from the centre of this Ball object to the
                    centre of the other Ball object at the time at
                    which the collision occurs.
                    Specified in terms of the components of the basis
                    vectors in the simulation's distance units.
            v1_zmf (n-tuple of real numeric types where n is the
                    number of spatial dimensions of the simulation):
                    The previously calculated velocity vector of this
                    Ball object in the zero momentum frame of this and
                    the other Ball object at the time at which the
                    collision occurs.
                    Specified in terms of the components of the basis
                    vector in the simulation's velocity units.
        
        Returns:
        None
        """
        self._updateTime0(t_collide)
        other._updateTime0(t_collide)
        v1 = self._v0
        v2 = other._v0
        mult = -self.m / other.m
        v2_zmf = tuple(mult * x for x in v1_zmf)
        rad_sum_sq = (self.radius + other.radius) ** 2
        dv1_sz = 2 * sum(x * y for x, y in\
                zip(contact_displ_vec, v1_zmf)) / rad_sum_sq
        self._v0 = tuple(x - dv1_sz * y for x, y in\
                zip(v1, contact_displ_vec))
        dv2_sz = 2 * sum(x * y for x, y in\
                zip(contact_displ_vec, v2_zmf)) / rad_sum_sq
        other._v0 = tuple(x - dv2_sz * y for x, y in\
                zip(v2, contact_displ_vec))
        return
    
    def isOutsideBox(self) -> Tuple[int]:
        """
        Checks whether any Ball object is completely contained within
        the box. If not, returns details regarding this (see below).
        
        If a Ball object is initially placed entirely inside the box,
        there should never be a point in time at which any part of it
        extends outside the box. This method can therefore act as a
        way of identifying that the collision detection and/or
        resolution for collisions between Ball objects and the walls of
        the box is not functioning correctly.
        
        Returns:
        Tuple, which:
        - If the Ball object has been identified as not being entirely
           within the box, is a 2-tuple which for one of the walls
           beyond which it extends beyond:
           - Index 0 contains a non-negative integer representing the
              index of the basis vector of the simulation which is
              normal to the wall.
           - Index 1 contains the integer 0 or 1, to specify to which
              of the two walls normal to the basis vector identified in
              Index 0 is being referred, with 0 representing that the
              wall in question is that from which the basis vector
              points into the box (referred to as the near wall), and 1
              representing that the wall in question is that from which
              the basis vector points out of the box (referred to as
              the far wall).
        - Otherwise is an empty tuple.
        """
        for axis_idx in range(self.n_dims):
            r = self.r[axis_idx]
            if self.centre_ranges[axis_idx][0] > r:
                return axis_idx, 0
            elif self.centre_ranges[axis_idx][1] < r:
                return axis_idx, 1
        return ()
    
    def calculateKineticEnergy(self) -> Real:
        """
        Calculates the current translational kinetic energy of the Ball
        object.
        
        Returns:
        Real numeric value giving the translational kinetic energy of
        the Ball object at the time represented by attribute t in
        terms of the simulation's energy units.
        """
        return 0.5 * self.m * sum(x ** 2 for x in self.v)
    
    def calculatePotentialEnergy(self) -> Real:
        """
        Calculates the current gravitational potential energy of the
        Ball object.
        The gravitational potential energy is measured relative to the
        spatial origin of the simulation (i.e. if the centre of the
        ball is at the spatial origin of the simulation, then the
        gravitational potential energy is defined to be zero).
        
        Returns:
        Real numeric value giving the gravitational potential energy of
        the Ball object at the time represented by attribute t in terms
        of the simulation's energy units.
        """
        return -sum(self.m * a * r for a, r in zip(self.g, self.r))
    
    def calculateMechanicalEnergy(self) -> Real:
        """
        Calculates the current mechanical energy (the translational
        kinetic energy plus the gravitational potential energy) of the
        Ball object.
        The gravitational potential energy is measured relative to the
        spatial origin of the simulation (i.e. if the centre of the
        ball is at the spatial origin of the simulation, then the
        gravitational potential energy is defined to be zero).
        
        Returns:
        Real numeric value giving the mechanical energy of the Ball
        object at the time represented by attribute t in terms of the 
        simulation's energy units.
        """
        return self.calculateKineticEnergy() +\
                self.calculatePotentialEnergy()

class MultiBallSimulation:
    """
    Class representing simulations in real n-dimensional space (where
    n is a strictly positive integer) with the following properties:
    - The simulation is contained within a finite, static
       n-hyperrectangular box
    - The box is considered to be in a uniform gravitational field
    - Objects in the simulation move according to Newtonian mechanics
    - The objects in the simulation and the walls of the box are
       assumed to be infinitely rigid and may not overlap with each
       other. Furthermore, any collision between a pair of objects
       in the simulation or between an object in the simulation and a
       wall are perfectly elastic
    - The box has infinite mass, so that the box remains in a
       constant state of motion (in the reference frame used, static)
       regardless of any collision between any ball any wall that
       may occur.
    - Other than when undergoing a collision, there are no forces
       between any pair of objects or between any object and any wall.
    - The box contains a number of n-dimensional balls, which are
       the only objects within the box. Contact between two balls or
       between a ball and the wall is frictionless (so the state of
       rotation of each ball does not change during from its initial
       state and does not influence the translational motion of the
       ball or any collisions, and is therefore ignored)
    - Each n-dimensional ball when introduced into the simulation
       is given a defined mass and radius (which are constant
       throughout the rest of the simulation) as well as an initial
       position and velocity. The position must be specified such
       that the ball is entirely within the box and it does not
       overlap with any other ball already present in the simulation.
    
    Each ball in the simulation is represented by an instance of the
    Ball class, and is the set of points in the n-dimensional space
    strictly less than a given distance (the radius of the ball) from a
    point in space (the cetnre of the ball), with the ball as a whole
    having a fixed radius and a fixed mass (neither of which change
    during the simulation). A ball being completely within the box
    means that the set of points of which it is comprised are all
    inside the box, and a pair of balls overlapping means that the
    intersection of the two sets of points of those balls is not empty.
    
    An elastic collision is one in which the sum of momentum and
    kinetic energy of the bodies involved immediately after the
    collision are the same as before the collision (recalling that the
    box is assumed to have infinite mass). Furthermore, the collisions
    are assumed to be instantaneous and with a change in the velocity
    vector which in the case of collision with a wall, is a vector
    normal to the wall at that point and in the case of a collision
    with another ball is a non-negative real scalar multiple of the
    vector from the centre of the ball being collided with to the
    centre of the ball in question at the time of the collision.
    
    In the absence of collisions, the motion of a ball is that of
    a rigid body under constant acceleration by a gravitational field
    as specified by g (see below). Outside of collisions, there is no
    force between balls or between any ball and any wall of the box.
    
    Mass, distance, and time are measured in terms of arbitrary
    but consistent units, with other units being derived from these,
    for example a speed/velocity unit being a unit of distance
    divided by a unit of time an acceleration unit being a unit of
    distance divided by a squared unit of time, a momentum unit being
    a unit of mass times a unit of velocity and an energy unit being
    a unit of mass being a unit of mass times a unit of velocity
    squared. We refer to these as the simulation's units.
    
    To describe vectors relating to the n-dimensional space (e.g.
    position vectors, displacement vectors, velocity vectors,
    momentum vectors, etc.), a consistent orthonormal basis is
    used, defined such that each basis vector is normal to a wall of
    the n-hyperrectangular box (in fact, each are normal to exactly 2
    walls of the box). These basis vectors are arbitrarily but
    consistently labelled with a different integer from 0 to (n - 1)
    (the label also referred to as the index of the corresponding
    basis vector. As an orthonormal basis, each basis vector has length
    1. A given vector relating to the n-dimensional space is
    represented as a 0-indexed n-tuple of real numeric values such that
    the value at a given index idx of the n-tuple is the numerical
    value of the signed projection (equivalently, the dot product) of
    that vector onto the basis vector labelled idx in terms of the
    corresponding units of the simulation (e.g. the simulation's
    distance units for position and displacement vectors, the
    simulation's speed units for velocity vectors, etc.). The values
    in this n-tuple are referred to as the components of the vector,
    with each component labelled by its index in the n-tuple (which
    by definition is equivalent to the label of the basis vector
    used).
    
    The different wall orientations are specified in terms of the
    label of the basis vector normal to the wall orientation. In
    order to distinguish between the two walls of the box of each
    orientation, we observe that for one of the walls, this basis
    vector points from that wall into the box and for the other it
    points from that wall away from the box. We refer to the former as
    the near wall and the latter as the far wall for that wall
    orientation.
    
    Finally, the time in the simulation as a whole is given as the time
    interval (in the simulation's time units) since the start of the
    simulation. This is referred to as the simulation's time measure.
    Note that by this definition, the time measure at start of the
    simulation is 0. Similarly, position vectors are defined to be the
    displacement from a point in space, which is chosen to be the
    intersection of all the near walls of the box (as defined above).
    We refer to this point as the spatial origin of the simulation.
    For example, for n = 2 in the standard Cartesian x-y plane, with
    the spatial origin at the origin of the x-y plane, the origin is at
    the lower left corner of the box.
    As such, for any position vector representing a point strictly
    within the box, every component of the vector in terms of the
    simulation's spatial basis is strictly positive.
    
    Initialisation args:
        
        Required positional:
        
        box_dims (n-tuple of strictly positive real numeric values):
                Specifies the value of the attribute box_dims, giving
                the dimensions of the n-hyperrectangular box in which
                the simulation is contained (i.e. for each basis
                vector in order, the distance between the two walls of
                the box that are normal to that basis vector), in terms
                of the simulation's distance units.
                
        Optional named:
        
        g (real numeric value or n-tuple of real numeric values):
                Specifies the uniform gravitational field of the
                simulation, in terms of the simulation's acceleration
                units.
                If given as a single real numeric value, then
                the gravitational field is orientated in the direction
                of the final basis vector (that with the largest index)
                of the simulation, with magnitude equal to the absolute
                value of the number given. A positive sign of that
                number signifyies that the field is parallel to that
                basis vector while a negative sign of that number
                signifyies that the field is antiparallel to that basis
                vector.
                Otherwise, this gives the components of the uniform
                gravitational field in terms of the basis vectors.
            Default: 0
    
    Attributes:
    
        Fixed (i.e. set at instance creation and unchanged through the
            lifetime of the object instance):
            
        n_dims (strictly positive int): The number of dimensions of
                the simulation (equal to the corresponding attribute
                of the attribute sim).
        g (n-tuple of real numeric values): Vector representing the
                uniform gravitational field in terms of the
                simulation's acceleration units.
        box_dims (n-tuple of strictly positive real numeric values):
                The dimensions of the n-hyperrectangular box in which
                the simulation is contained (i.e. for each basis
                vector in order, the distance between the two walls of
                the box that are normal to that basis vector), in terms
                of the simulation's distance units.
        box_interior_ranges (n-tuple of 2-tuples of real numeric
                values): The open ranges such that a position vector
                represents a point inside the n-hyperrectangular box if
                and only if every one of its components is within
                the corresponding open range.
        
        Updateable (i.e. can be updated by the user at any point
            during the lifetime of the instance):
            
        t (real numeric value): The current time for the simulation
                in terms of the simulation's time measure.
        
        Derived (i.e. set at instance creation and updated based on
            calculations made using other attribute values):
            
        balls (list of Ball objects): The Ball objects in the
                simulation.
        ball_states (list of integers): A list of the same length as
                the attribute balls. Each entry contains a non-negative
                integer giving the state of the corresponding Ball
                object in the balls attribute.
                The state of a Ball object in the simulation is the
                total number of collisions (with both other objects and
                walls) has experienced up to the current time in the
                simulation.
        balls_collision_heaps (list of lists, the latter of which are
                arranged as a min-heaps by the heapq package): A list
                of the same length as the attribute balls. The entry
                at a given index contains a list arranged as a min-heap
                whose entries contain details of the collisions the
                Ball object at the same index in the attribute balls
                with other Ball object that at the current time (with
                the Ball objects following their current trajectories)
                are on course to occur, arranged by the time the
                collision is projected to take place.
                For a collision currently on course to occur between
                the Ball objects with indices idx1 and idx2, the
                details of the collision is stored in either the
                min-heap at index idx1 or the min-heap at index idx2
                of this attribute. Which it is stored in does not
                matter, but it must be stored in exactly one of these
                (not neither or both).
                If on this occasion, it is stored in the min-heap at
                idx1, then the details of the collision are stored in
                this heap as a 7-tuple with the following structure:
                - Index 0 contains a real numeric value representing
                   the time the collision is projected to occur (in the
                   simulation's time measure).
                - Index 1 contains an n-tuple of real numeric values
                   representing the displacement vector from the centre
                   of the Ball object with index idx1 to the centre of
                   the Ball object with index idx2 at the projected
                   point of collision.
                - Indices 2 and 3 contain non-negative integers idx1
                   and the current state of the Ball object with index
                   idx1 (i.e. the total number of collisions that Ball
                   object has previously experienced over the course
                   of the simulation at the current time, as recorded
                   in the attribute ball_states) respectively.
                - Index 4 contains an n-tuple of real numeric values
                   representing the velocity of the centre of the Ball
                   object with index idx1 at the projected point of
                   collision in the zero momentum frame of the two Ball
                   objects (i.e. the frame of reference moving in such
                   a way that the total momentum of all bodies involved
                   is zero).
                - Indices 5 and 6 contain non-negative integers idx2
                   and the current state of the Ball object with index
                   idx2 respectively.
                As previously mentioned, the min-heap is arranged based
                on the time the corresponding collision is due to
                occur, so that the top of the heap (at index 0 in the
                list representing the heap) is guaranteed to have the
                (joint) smallest (and so soonest) time for the
                projected collision it describes out of all entries in
                the heap.
    
    Methods:
        (For full description, see documentation of the method itself)
        
        progressTime(): Progresses the simulation up to a specified
                time.
        calculateTotalKineticEnergy(): Calculates the total
                translational kinetic energy of the objects in the
                simulation.
        calculateTotalPotentialEnergy(): Calculates the total
                gravitational potential energy of the objects in the
                simulation.
        calculateTotalMechanicalEnergy(): Calculates the total
                mechanical energy of the objects in the simulation.
        detectAnyBallOutsideBox(): Checks whether there are any Ball
                objects in the simulation that are not entirely
                inside the n-hyperrectangular box. Acts as a way
                of identifying whether collision detection and/or
                resolution for collisions between objects in the
                simulation and walls is not working correctly.
        detectAnyBallsOverlap(): Checks whether there is any overlap
                between any pair of Ball objects in the simulation.
                Acts as a way of identifying whether collision
                detection and/or resolution for collisions between
                objects in the simulation and walls is not working
                correctly.
        anyOverlapMessage(): Checks whether there are any objects
                in the simulation that are not entirely inside the
                n-hyperrectangular box and whether there exists a
                pair of Ball objects in the simulation that overlap
                with each other. If so, provides a message giving
                details of one such occurrence. Acts to combine the
                checks of detectAnyBallOutsideBox() and
                detectAnyBallsOverlap().
    """
    def __init__(self, box_dims: Tuple[Real],\
            g: Union[Real, Tuple[Real]]=0):
        
        self._box_dims = box_dims
        self._box_interior_ranges = tuple((0, x) for x in box_dims)
        self._g = g if hasattr(g, "__getitem__") else\
                tuple([0] * (self.n_dims - 1) + [g])
        
        self.t = 0
        
        self.balls = []
        self.ball_states = []
        
        self._t_ref = float("inf")
    
    @property
    def box_dims(self):
        return self._box_dims
    
    @property
    def box_interior_ranges(self):
        return self._box_interior_ranges
    
    @property
    def n_dims(self):
        return len(self._box_dims)
    
    @property
    def g(self):
        return self._g
    
    def addBall(self, m: Real, radius: Real,\
            r0: Tuple[Real], v0: Tuple[Real],\
            check_overlap: bool=True,\
            balls_t_updated: bool=False) -> bool:
        """
        Adds a Ball object to the simulation at the current time,
        representing a ball with mass in n-dimensional space (where
        n is the number of spatial dimensions of the simulation)
        which collides perfectly elastically with both walls and other
        Ball objects.
        If check_overlap given as True then before adding the Ball
        checks that it does not overlap any other objects in the
        simulation and is completely inside the box. If this check
        is failed, then the Ball object is not added to the simulation.
        
        In the following:
        - n represents the number of spatial dimensions of the
            simulation (equal to the attribute n_dims)
        - The current time of the simulation refers to the time
            corresponding to the simulation's time measure having
            a numerical value equal to the value of attribute t
            at the point this method is called.
        - Vectors are representeded using their components in the
            simulation's orthonormal basis as an n-tuple of real
            numeric values, with their units being the corresponding
            unit of the simulation (e.g. the simulation's distance
            units for position and displacement vectors, the
            simulation's speed units- where one speed unit is equal to
            one distance unit divided by one time unit- for velocity
            vectors, etc.).
        - The origin used for all position vectors is the spatial
            origin of the simulation (as defined in the documentation
            for MultiBallSimulation)
        
        Args:
            Required positional:
            m (strictly positive real numeric value): The mass of the
                    object in terms of the simulation's mass units.
            radius (strictly positive real numeric value): The radius
                    of the new Ball object in terms of the simulation's
                    mass units.
            r0 (n-tuple of real numeric values): The position vector of
                    the centre of the new Ball object at the current
                    time.
            v0 (n-tuple of real numeric values): The velocity vector of
                    the centre of the new Ball object at the current
                    time.
            
            Optional named:
            
            check_overlap (bool): If True then checks to ensure that
                    the new Ball object being added would be completely
                    within the box and would not overlap with any
                    object previously added to the simulation. If
                    this check is performed and either of those
                    conditions is not satisfied, then the new Ball
                    object is not added to the simulation.
                    This should only be given as False in a situation
                    where both of the conditions are guaranteed to
                    be satisfied. Otherwise, the simulation may not
                    function correctly.
                Default: True
            balls_t_updated (bool): If True and check_overlap is also
                    True, when checking for overlaps between the new
                    Ball object and any of the previously added Ball
                    objects (contained in the attribute balls), it is
                    assumed that all of those previously added Ball
                    objects have the same attribute t as that of the
                    simulation (i.e. have the same current time).
                    Otherwise, this is not assumed, and (if
                    check_overlap is True) the attribute t of those
                    previously added Ball objects is progressed to
                    match that of the simulation.
                Default: False
        
        Returns:
        Boolean (bool), True if the Ball object was added to the
        simulation, False if not (which only occurs when check_overlap
        is given as False and either the definition of the new Ball
        object does not place it completely within the box or it
        overlaps with an object previously added to the simulation).
        """
        ball = Ball(self, m, radius, r0, v0, t0=self.t)
        if check_overlap:
            if not balls_t_updated:
                self._updateBallsTime()
            if ball.isOutsideBox() or\
                    self._detectBallsOverlap(ball, start_idx=0):
                #print("failed check")
                return False
            #print("passed check")
        self.balls.append(ball)
        self.ball_states.append(0)
        return True
    
    def _ballsCollisionHeapEntry(self, idx1: int, idx2: int)\
            -> Tuple[Union[Real, Tuple[Real]]]:
        """
        Finds the next time (if any) two specific Ball objects in the
        simulation are due to collide with each other and provides
        details of this projected collision in a format that can
        be inserted directly into the appropriate min-heap in the
        attribute balls_collision_heaps.
        
        In the following:
        - n represents the number of spatial dimensions of the
            simulation (equal to the attribute n_dims)
        - Vectors are represented using their components in the
            simulation's orthonormal basis as an n-tuple of real
            numeric values, with their units being the corresponding
            unit of the simulation (e.g. the simulation's distance
            units for position and displacement vectors, the
            simulation's speed units- where one speed unit is equal to
            one distance unit divided by one time unit- for velocity
            vectors, etc.).
        
        Args:
            Required positional:
            idx1 (int): Non-negative integer giving the index in
                    attribute balls for one of the pair of Ball
                    objects being examined. This is also the index
                    in the attribute balls_collision_heaps into
                    which the output of this method is intended
                    to be inserted
            idx2 (int): Non-negative integer giving the index in
                    attribute balls of the other of the pair of
                    Ball objects being examined.
        
        Returns:
        Tuple, which:
        - If the pair of Ball objects are currently on course to
           collide with each other, is 7-tuple where:
           - Index 0 contains a real numeric value representing the
              time the collision is projected to occur (in the
              simulation's time measure).
           - Index 1 contains an n-tuple of real numeric values
              representing the displacement vector from the centre
              of the Ball object with index idx1 to the centre of
              the Ball object with index idx2 at the projected
              point of collision.
           - Indices 2 and 3 contain non-negative integers idx1 and
              the current state of the Ball object with index idx1
              (i.e. the total number of collisions that Ball object has
              previously experienced over the course of the simulation
              at the current time, as recorded in the attribute
              ball_states) respectively.
           - Index 4 contains an n-tuple of real numeric values
              representing the velocity of the centre of the Ball
              object with index idx1 at the projected point of
              collision in the zero momentum frame of the two Ball
              objects (i.e. the frame of reference moving in such a
              way that the total momentum of all bodies involved is
              zero).
           - Indices 5 and 6 contain non-negative integers idx2 and
              the current state of the Ball object with index idx2
              respectively.
           This tuple is designed to be added directly to the min-heap
           at index idx1 of the attribute balls_collision_heaps.
        - Otherwise, is an empty tuple.
        
        """
        ball1, ball2 = self.balls[idx1], self.balls[idx2]
        ans = ball1.identifyOtherBallNextCollision(ball2)
        if ball1.t > self._t_ref:
            print(f"collision of balls {idx1} and {idx2}: {ans}")
        if not ans: return ()
        t2, contact_displ_vec, v1_zmf = ans
        return (t2, contact_displ_vec, idx1, self.ball_states[idx1],\
                v1_zmf, idx2, self.ball_states[idx2])
    
    def _initialiseBallsCollisionHeaps(self) -> None:
        """
        Completely sets (or resets) the attribute balls_collision_heaps
        based on the current trajectories of the objects in the
        simulation.
        
        Returns:
        None
        """
        res = {}
        n_balls = len(self.balls)
        for idx1 in range(n_balls):
            heap = []
            for idx2 in range(idx1 + 1, n_balls):
                ans = self._ballsCollisionHeapEntry(idx1, idx2)
                if ans: heap.append(ans)
            if heap:
                heapq.heapify(heap)
                res[idx1] = heap
        self.balls_collision_heaps = res
        return
    
    def _resetBallsCollisionHeap(self, idx: int, excl: Set[int])\
            -> bool:
        """
        Completely resets the min-heap at index idx of the attribute
        balls_collision_heaps based on the current trajectories of the
        object in the simulation, by identifying which other objects
        in the simulation are currently on course to collide with
        the Ball object at index idx of the attribute balls.
        
        This is designed to be used after a change in trajectory of the
        corresponding Ball object (e.g. immediately after a collision
        with another object in the simulation or a wall).
        
        Args:
            Required positional:
            idx (int): Non-negative integer giving the index in
                    attribute balls_collision_heaps of the min-heap to
                    be reset.
            excl (set of ints): A set of indices of the attribute balls
                    for which the next collisions between the Ball
                    object in question and the Ball objects
                    corresponding to those indices should not be
                    included in the min-heap.
                    This option exists to avoid the possibility of,
                    after a collision between two Ball objects, a
                    further collision between the two being included
                    in the min-heap in balls_collision_heaps for
                    both objects (and so being double-counted), by
                    including the index of the other Ball object
                    in this set during exactly one of the calls of
                    this method.
                    Note that with the current physics used, after
                    two Ball objects collide, they should never
                    immediately be on course to collide again.
                    Therefore, this step is currently not strictly
                    necessary, although it does save checking for a
                    projected collision which cannot occur (and so
                    saves a minimal amount of time) and also guards
                    against this causing errors in the simulation if
                    the physics model is changed.
        
        Returns:
        Boolean (bool), giving True if the reset min-heap is non-empty,
        otherwise False.
        """
        heap = []
        ball1 = self.balls[idx]
        if ball1.t > self._t_ref:
            print(f"resetting ball {idx} collision heap")
        excl_sorted = sorted(excl.union({idx}))
        rngs = []
        prev = 0
        for curr in excl_sorted:
            if curr == prev:
                continue
            rngs.append(range(prev, curr))
            prev = curr + 1
        n_balls = len(self.balls)
        if prev != n_balls:
            rngs.append(range(prev, n_balls))
        for idx2 in itertools.chain(*rngs):
            if idx2 in excl: continue
            if ball1.t > self._t_ref:
                print(idx2)
            ans = self._ballsCollisionHeapEntry(idx, idx2)
            if ans: heap.append(ans)
        if heap:
            heapq.heapify(heap)
            self.balls_collision_heaps[idx] = heap
            return True
        return False
    
    def _updateBallsCollisionHeap(self, idx: int, t_max: Real)\
            -> bool:
        """
        Updates the min-heap at index idx of the attribute
        balls_collision_heaps by repeatedly removing the item at the
        top of the heap as long as its time does not exceed that
        represented by t_max and it no longer represents a collision
        that is on course to occur (due to the trajectory of either
        Ball object having changed since that item was added to the
        min-heap).
        
        Args:
            idx (int): Non-negative integer giving the index in the
                    attribute balls_collision_heaps containing the
                    min-heap to be updated.
            t_max (real numeric value): The time (in terms of the
                    simulation's time measure) such that if and when
                    the item at the top of the min-heap has a time
                    exceeds this, it is not removed (even if it
                    no longer represents a collision that is on
                    course to occur).
        
        Returns:
        Boolean (bool), giving True if after being updated the min-heap
        at index idx of the attribute balls_collision_heaps is
        non-empty and the item at the top of the heap has a time no
        greater than that represented by t_max (and so represents a
        collision that is still on course to occur), otherwise False.
        """
        bc_heap = self.balls_collision_heaps.get(idx, [])
        while bc_heap:
            tup = bc_heap[0]
            if tup[0] > t_max: return ()
            idx2 = tup[5]
            if tup[3] == self.ball_states[idx] and\
                    tup[6] == self.ball_states[idx2]:
                return True#(tup[0], idx2)
            heapq.heappop(bc_heap)
        return False#()
    
    def _constructGlobalBallsCollisionHeap(self, t_max: Real)\
            -> List[Tuple[Real]]:
        """
        Constructs a min-heap containing the details of the earliest
        collision(s) (if any) each Ball object in the simulation is
        currently on course to have with any other Ball object provided
        that this collision is projected to occur no later than the
        time represented by t_max.
        The min-heap is arranged based on the time the corresponding
        collision is due to occur, so that the top of the heap (at
        index 0 in the heap, as represented by a list) has the
        smallest (or joint smallest) time for the corresponding
        collision out of all entries in the heap.
        
        Note that this min-heap is guaranteed to contain the earliest
        collision each Ball object in the simulation is on course
        to be involved in (provided it is projected to occur no later
        than t_max), but due to the mechanism this heap is updated
        and interacts with the rest of the simulation, it may also
        contain details of projected collisions where for neither of
        the Ball objects involved is that collision their first.
        
        Further note that care has been taken to ensure that each
        projected collision is only included once and in particular,
        the min-heap constructed will not contain the same entry twice
        with the details of the two Balls involved swapping positions
        (which could give rise to significant issues for the
        simulation)
        
        Args:
            Required positional:
            t_max (real numeric value): The time (in terms of the
                    simulation's time measure) such that collisions
                    projected to occur after this time are not inserted
                    into the min-heap.
        
        Returns:
        List arranged as a min-heap by the heapq package. Each item
        in the list is a 5-tuple representing a collision between
        two Ball objects that is currently on course to happen, where:
        - Index 0 contains a real numeric value giving the time the
           projected collision is currently due to occur (in terms of
           the simulation's time measure)
        - Indices 1 and 2 contain non-negative integers giving the
           index in the attribute balls corresponding to one of the
           Ball objects involved in the collision (and the one whose
           min-heap in balls_collision_heaps contains the full details
           of the collision) and the current state of that Ball object
           (i.e. the total number of collisions that Ball object has
           previously experienced over the course of the simulation at
           the current time, as recorded in the attribute ball_states)
           respectively
        - Indices 3 and 4 similarly contain non-negative integers
           giving the index in the attribute balls corresponding to the
           other Ball object involved in the collision and the current
           state of that Ball object respectively
        The min-heap is arranged based on the times of the projected
        collisions (with the smallest and so soonest time at the top,
        of the heap, or equivalently at index 0 of the heap).
        """
        if not hasattr(self, "balls_collision_heaps"):
            self._initialiseBallsCollisionHeaps()
        
        heap = []
        for idx1 in list(self.balls_collision_heaps.keys()):
            if self._updateBallsCollisionHeap(idx1, t_max):
                tup = self.balls_collision_heaps[idx1][0]
                t, idx2 = tup[0], tup[5]
                heap.append((t, idx1, self.ball_states[idx1],\
                        idx2, self.ball_states[idx2]))
            elif not self.balls_collision_heaps[idx1]:
                self.balls_collision_heaps.pop(idx1)
        heapq.heapify(heap)
        return heap
    
    def _constructGlobalWallCollisionHeap(self, t_max: Real)\
            -> List[Tuple[Real]]:
        """
        Constructs a min-heap containing the details of the earliest
        collision (if any) each Ball object in the simulation is
        currently on course to have with a wall, provided that this
        collision is projected to occur no later than the time
        represented by t_max.
        The min-heap is arranged based on the time the corresponding
        collision is due to occur, so that the top of the heap (at
        index 0 in the heap, as represented by a list) has the
        smallest (or joint smallest) time for the corresponding
        collision out of all entries in the heap.
        
        Args:
            Required positional:
            t_max (real numeric value): The time (in terms of the
                    simulation's time measure) such that collisions
                    projected to occur after this time are not inserted
                    into the min-heap.
        
        Returns:
        List arranged as a min-heap by the heapq package. Each item
        in the list is a 3-tuple representing a collision between
        Ball object and a wall that is currently on course to happen,
        where:
        - Index 0 contains a real numeric value giving the time the
           projected collision is currently due to occur (in terms of
           the simulation's time measure)
        - Index 1 contains a non-negative integer giving the index in
           the attribute balls corresponding to the Ball object
           involved in the projected collision
        - Index 2 contains a non-negative integer giving the current
           state of the Ball object (i.e. the total number of
           collisions that Ball object has previously experienced
           over the course of the simulation at the current time, as
           recorded in the attribute ball_states)
        The min-heap is arranged based on the times of the projected
        collisions (with the smallest and so soonest time at the top,
        of the heap, or equivalently at index 0 of the heap).
        """
        heap = []
        for i, ball in enumerate(self.balls):
            nw_heap = ball.next_wall_heap
            if not nw_heap: continue
            t, dim_idx = nw_heap[0]
            if t > t_max: continue
            heap.append((t, i, self.ball_states[i]))
        heapq.heapify(heap)
        return heap
    
    def _updateGlobalBallsCollisionHeap(self,\
            gc_heap: List[Tuple[Real]], t_max: Real) -> bool:
        """
        Updates the min-heap gc_heap (used to identify the next
        collision between any pair of Ball objects in the simulation
        currently on course to occur) by repeatedly removing the item
        at the top of the heap as long as it no longer represents a
        collision that is on course to occur (due to the trajectory
        of either of the corresponding Ball objects having changed
        since that item was added to the min-heap), replacing the
        removed item in the min-heap if and as appropriate.
        
        Args:
            gc_heap (list of 5-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between pairs of Ball objects currently
                    on course to occur in the simulation no later than
                    the time t_max, with the min-heap organised based
                    on the projected times of the corresponding
                    collisions (with a collision projected to occur
                    the soonest or joint soonest at the top of the
                    heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalBallsCollisionHeap().
                    See the documentation of the former of this method
                    for more details regarding the construction of the
                    min-heap.
            t_max (real numeric value): The time (in terms of the
                    simulation's time measure) such that if and when
                    the item at the top of the min-heap has a time
                    exceeds this, it is not removed (even if it
                    no longer represents a collision that is on
                    course to occur).
        
        Returns:
        Boolean (bool), giving True if after being updated the min-heap
        gc_heap is non-empty and the item at the top of the heap has a
        time no greater than that represented by t_max (and so
        represents a collision that is still on course to occur),
        otherwise False.
        """
        if t_max > self._t_ref:
            print(f"Finding next ball collision")
        while gc_heap:
            t, idx1, state1, idx2, state2 = gc_heap[0]
            if self.ball_states[idx1] == state1 and\
                    self.ball_states[idx2] == state2:
                return True
            bc_heap = self.balls_collision_heaps.get(idx1, [])
            if not bc_heap or bc_heap[0][2] != idx1 or\
                    bc_heap[0][3] != state1 or\
                    bc_heap[0][5] != idx2 or\
                    bc_heap[0][6] != state2:
                heapq.heappop(gc_heap)
                continue
            
            if t_max > self._t_ref:
                print(f"Soonest gc_heap entry at t = {t}, "\
                        f"idx1 = {idx1}")
                print(f"Soonest bc_heap entry for ball {idx1} = "\
                        f"{bc_heap[0]}")
                print(f"ball states = {self.ball_states}")
            idx2 = bc_heap[0][5]
            if len(bc_heap) == 1:
                self.balls_collision_heaps.pop(idx1)
                heapq.heappop(gc_heap)
                continue
            heapq.heappop(bc_heap)
            if self._updateBallsCollisionHeap(idx1, t_max):
                t, idx2 = bc_heap[0][0], bc_heap[0][5]
                heapq.heappushpop(gc_heap,\
                        (t, idx1, self.ball_states[idx1],\
                        idx2, self.ball_states[idx2]))
            else: heapq.heappop(gc_heap)
                
        return False
    
    def _updateGlobalWallCollisionHeap(self,\
            gnw_heap: List[Tuple[Real]]) -> bool:
        """
        Updates the min-heap gnw_heap (used to identify the next
        collision between a Ball object in the simulation and a wall
        currently on course to occur) by repeatedly removing the item
        at the top of the heap as long as it no longer represents a
        collision that is on course to occur (due to the trajectory of
        the corresponding Ball object having changed since that item
        was added to the min-heap).
        
        Args:
            Required positional:
            gnw_heap (list of 3-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between a Ball object and a wall
                    currently on course to occur in the simulation no
                    later than the time t_max, with the min-heap
                    organised based on the projected times of the
                    corresponding collisions (with a collision
                    projected to occur the soonest or joint soonest at
                    the top of the heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalWallCollisionHeap(). See
                    the documentation of of this method for more
                    details regarding the construction of the min-heap.
        
        Returns:
        Boolean (bool), giving True if after being updated the min-heap
        gnw_heap is non-empty (and so the item at the top of the heap
        represents a collision that is still on course to occur),
        otherwise False.
        """
        while gnw_heap:
            t, i, state = gnw_heap[0]
            if state == self.ball_states[i]:
                return True
            heapq.heappop(gnw_heap)
        return False
    
    def _updateBallsStateAfterBallsCollision(self, idx1: int,\
            idx2: int, t_max: Real, gc_heap: List[Tuple[Real]],\
            gnw_heap: List[Tuple[Real]]) -> None:
        """
        Implements the necessary changes to be made immediately
        following a collision between two Ball objects in the
        simulation. Specifically:
        - Increments the state of the two Ball objects involved
           by 1.
        - Resets the attribute next_wall_heap of those two Ball objects
           (using the method initialiseNextWallHeap() of each Ball
           object) and the attribute balls_collision_heaps of the
           simulation (using the method _resetBallsCollisionHeap())
           for those two Ball objects to reflect their respective
           changes in trajectory and therefore the collisions each
           object is now on course for resulting from the collision.
        - Updates the heaps gnw_heap and gc_heap to similarly reflect
           the effect these changes in trajectory have on the upcoming
           collisions with walls and between Ball objects respectively
           in the simulation as a whole.
        It is assumed that the value of the attribute t (representing
        the current time in terms of the simulation's time measure) for
        the simulation and the two Ball objects has already been set to
        represent the time of the collision. Furthermore, the change in
        velocity of the two Ball objects resulting from the collision
        is assumed to already have been applied (by the appropriate use
        of the method progressToNextOtherBallCollision() for one of the
        two Ball objects).
        
        Note that the choice of which of the Ball objects is
        represented by the index idx1 (as opposed to idx2) does not
        matter.
        
        Args:
            Required positional:
            idx1 (int): Non-negative integer giving the index in the
                    attribute balls of one of the two Ball objects
                    involved in the collision.
            idx2 (int): Non-negative integer giving the index in the
                    attribute balls of the other of the two Ball
                    objects involved in the collision.
            t_max (real numeric value):
            gc_heap (list of 5-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between pairs of Ball objects currently
                    on course to occur in the simulation no later than
                    the time t_max, with the min-heap organised based
                    on the projected times of the corresponding
                    collisions (with a collision projected to occur
                    the soonest or joint soonest at the top of the
                    heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalBallsCollisionHeap() and
                    kept updated using the method
                    _updateGlobalBallsCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the construction of the
                    min-heap.
            gnw_heap (list of 3-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between a Ball object and a wall
                    currently on course to occur in the simulation no
                    later than the time t_max, with the min-heap
                    organised based on the projected times of the
                    corresponding collisions (with a collision
                    projected to occur the soonest or joint soonest at
                    the top of the heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalWallCollisionHeap() and
                    kept updated using the method
                    _updateGlobalWallCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the construction of the
                    min-heap.
        
        Returns:
        None
        """
        if t_max > self._t_ref:
            print(f"Pre, t = {t_max}, ball {idx1}: "\
                    f"{self.balls_collision_heaps.get(idx1, [])}\n ball "\
                    f"{idx2}: {self.balls_collision_heaps.get(idx2, [])}"\
                    f"\n gc_heap: {gc_heap}")
        for idx in (idx1, idx2):
            self.ball_states[idx] += 1
            ball = self.balls[idx]
            ball.initialiseNextWallHeap()
            if ball.next_wall_heap:
                t = ball.next_wall_heap[0][0]
                if t <= t_max:
                    heapq.heappush(gnw_heap,\
                            (t, idx, self.ball_states[idx]))
        excl = set()
        for idx in (idx2, idx1):
            if self._resetBallsCollisionHeap(idx, excl):
                tup = self.balls_collision_heaps[idx][0]
                if tup[0] <= t_max:
                    ans = (tup[0], tup[2], tup[3], tup[5], tup[6])
                    heapq.heappush(gc_heap, ans)
            # Future-proofing against changes in the physics used
            # (e.g. a non-uniform gravitational field) giving rise to
            # the possibility of two Ball objects being on course to
            # collide with each other immediately following a collision
            # between the pair (which is currently impossible). The
            # following action guards against double-counting of the
            # new projected collision (i.e. its inclusion in both
            # balls_collision_heaps[idx1] and
            # balls_collision_heaps[idx2], guaranteeing that it cannot
            # appear in the latter).
            excl.add(idx)
        if t_max > self._t_ref:
            bc_heap1 = self.balls_collision_heaps.get(idx1, [])
            bc_heap2 = self.balls_collision_heaps.get(idx2, [])
            print(f"Post, t = {t_max}, ball {idx1}: {bc_heap1}"\
                    f"\n ball {idx2}: {bc_heap2}\n gc_heap: {gc_heap}")
        return
    
    def _updateBallStateAfterWallCollision(self, idx: int, t_max: Real,\
            gc_heap: List[Tuple[Real]],\
            gnw_heap: List[Tuple[Real]]) -> None:
        """
        Implements the necessary changes to be made immediately
        following a collision between a Ball object and a wall in the
        simulation. Specifically:
        - Increments the state of the Ball object involved by 1.
        - Updates the attribute next_wall_heap of the Ball object
           (using its method initialiseNextWallHeap()) and resets the
           attribute balls_collision_heaps of the simulation (using
           the method _resetBallsCollisionHeap()) for the Ball object
           to reflect its change in trajectory and therefore the
           collisions it is now on course for resulting from the
           collision.
        - Updates the heaps gnw_heap and gc_heap to similarly reflect
           the effect these changes in trajectory have on the upcoming
           collisions with walls and between Ball objects respectively
           in the simulation as a whole.
        It is assumed that the value of the attribute t (representing
        the current time in terms of the simulation's time measure) for
        the simulation and the Ball object has already been set to
        represent the time of the collision. Furthermore, the change in
        velocity of the Ball object resulting from the collision is
        assumed to already have been applied (by the appropriate use
        of the Ball object's method progressToNextWallCollision()).
        
        Args:
            Required positional:
            idx (int): Non-negative integer giving the index in the
                    attribute balls of the Ball object involved in the
                    collision.
            t_max (real numeric value): The latest time (in terms of
                    the simulation's time measure) of new projected
                    collisions for which corresponding items are
                    added to gnw_heap or gc_heap.
            gc_heap (list of 5-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between pairs of Ball objects currently
                    on course to occur in the simulation no later than
                    the time t_max, with the min-heap organised based
                    on the projected times of the corresponding
                    collisions (with a collision projected to occur
                    the soonest or joint soonest at the top of the
                    heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalBallsCollisionHeap() and
                    kept updated using the method
                    _updateGlobalBallsCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the construction of the
                    min-heap
            gnw_heap (list of 3-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between a Ball object and a wall
                    currently on course to occur in the simulation no
                    later than the time t_max, with the min-heap
                    organised based on the projected times of the
                    corresponding collisions (with a collision
                    projected to occur the soonest or joint soonest at
                    the top of the heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalWallCollisionHeap() and
                    kept updated using the method
                    _updateGlobalWallCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the construction of the
                    min-heap.
        
        Returns:
        None
        """
        self.ball_states[idx] += 1
        ball = self.balls[idx]
        if ball.next_wall_heap:
            t = ball.next_wall_heap[0][0]
            if t <= t_max:
                heapq.heappush(gnw_heap,\
                        (t, idx, self.ball_states[idx]))
        if self._resetBallsCollisionHeap(idx, set()):
            tup = self.balls_collision_heaps[idx][0]
            #print(t, t_max)
            if tup[0] <= t_max:
                ans = (tup[0], tup[2], tup[3], tup[5], tup[6])
                heapq.heappush(gc_heap, ans)
        return
    
    def _progressToNextCollision(self, t_max: Real,\
            gc_heap: List[Tuple[Real]],\
            gnw_heap: List[Tuple[Real]]) -> bool:
        """
        Progresses the simulation until immediately after the next
        collision due to occur in the simulation (either a collision
        between two objects or between an object and a wall) as long
        as this occurs no later than the time represented by t_max.
        
        Args:
            t_max (real numeric value): The latest time (in terms of
                    the simulation's time measure) of new projected
                    collisions for which corresponding items are
                    added to gnw_heap or gc_heap.
            gc_heap (list of 5-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between pairs of Ball objects currently
                    on course to occur in the simulation no later than
                    the time t_max, with the min-heap organised based
                    on the projected times of the corresponding
                    collisions (with a collision projected to occur
                    the soonest or joint soonest at the top of the
                    heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalBallsCollisionHeap() and
                    kept updated using the method
                    _updateGlobalBallsCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the min-heap.
            gnw_heap (list of 3-tuples arranged as a min-heap by the
                    heapq package): Min-heap storing details of
                    collisions between a Ball object and a wall
                    currently on course to occur in the simulation no
                    later than the time t_max, with the min-heap
                    organised based on the projected times of the
                    corresponding collisions (with a collision
                    projected to occur the soonest or joint soonest at
                    the top of the heap).
                    This min-heap should have been constructed using
                    the method _constructGlobalWallCollisionHeap() and
                    kept updated using the method
                    _updateGlobalWallCollisionHeap(). See the
                    documentation of the former of these methods for
                    more details regarding the construction of the
                    min-heap.
        
        Returns:
        Boolean (bool) which if True signifies that the method has
        progressed the simulation to immediately after the next
        collision, otherwise False (which additionally signifies that
        the next collision in the simulation will occur after the time
        represented by t_max).
        """
        b1 = self._updateGlobalBallsCollisionHeap(gc_heap, t_max)
        b2 = self._updateGlobalWallCollisionHeap(gnw_heap)
        if t_max > self._t_ref:
            print("Finding next collision")
            print(f"Global balls collision heap: {gc_heap}")
            print(f"Global walls collision heap: {gnw_heap}")
            print("Balls collision heaps: "\
                    f"{self.balls_collision_heaps}")
        if not b1 and not b2: return False
        if not b1 or (b2 and gc_heap[0][0] > gnw_heap[0][0]):
            i = heapq.heappop(gnw_heap)[1]
            #self.applyNextWallCollision(i, t_max, gc_heap, gnw_heap)
            self.balls[i].progressToNextWallCollision()
            if t_max > self._t_ref:
                print(f"Applying collision between ball {i} and wall "\
                        f"at t = {self.balls[i].t}")
            self._updateBallStateAfterWallCollision(i, t_max, gc_heap,\
                    gnw_heap)
            return True
        i1 = heapq.heappop(gc_heap)[1]
        #self.applyNextBallCollision(i1, t_max, gc_heap, gnw_heap)
        if t_max > self._t_ref:
            print(f"gc_heap = {gc_heap}")
            print(f"ball collision heap for ball {i1} = "\
                    f"{self.balls_collision_heaps[i1]}")
            print(f"ball states = {self.ball_states}")
        t, contact_displ_vec, _, _, v1_zmf, i2, _ =\
                heapq.heappop(self.balls_collision_heaps[i1])
        if not self.balls_collision_heaps[i1]:
            self.balls_collision_heaps.pop(i1)
        ball1, ball2 = self.balls[i1], self.balls[i2]
        if t_max > self._t_ref:
            print(f"Applying collision between balls {i1} and {i2} "\
                    f"at t = {t}")
            d_sq = sum((x - y) ** 2 for x, y in zip(ball1.r, ball2.r))
            rad_sq = (ball1.radius + ball2.radius)
            print(f"For balls {i1} and {i2}, squared centre distance "\
                    f"= {d_sq}, squared radius sum = {rad_sq}")
        self.balls[i1].progressToNextOtherBallCollision(\
                self.balls[i2], t, contact_displ_vec, v1_zmf)
        self._updateBallsStateAfterBallsCollision(\
                i1, i2, t_max, gc_heap, gnw_heap)
        return True
    
    def _updateBallsTime(self) -> None:
        """
        Sets the current time of all of the Ball objects (as
        represented by attribute t in the simulation's time measure)
        to equal that of the simulation.
        
        Returns:
        None
        """
        for ball in self.balls:
            ball.t = self.t
    
    def progressTime(self, dt: Real, check_overlap: bool=True) -> int:
        """
        Progresses the simulation by the time interval dt from the
        current time (attribute t). Accounts for the trajectories of
        the objects in the simulation as well as any collisions between
        two objects and between an object and a wall that occur during
        this time interval.
        
        Args:
            Required positional:
            dt (real numeric value): The time interval (in terms of the
                    simulation's time units) the simulation is to be
                    progressed.
            
            Optional named:
            check_overlap (bool): If True then after the simulation has
                    been progressed, inspects each pair of objects
                    in the simulation to see if any pairs overlap with
                    each other. If so, prints to console the details of
                    one of the pairs of objects found to overlap.
                    During the simulation, (as long as there is no
                    overlap between any pair of objects when the
                    objects are first added to the simulation) no pair
                    of objects should ever overlap with each other and
                    so for a simulation with valid initial conditions,
                    such a message being printed signifies an error
                    in detection and/or resolution of collisions
                    between objects in the simulation.
                Default: True
        
        Returns:
        Non-negative integer (int) representing the number of
        collisions that occurred (both those between two objects and
        between an object and a wall) in the simulation during this
        time interval.
        """
        t2 = self.t + dt
        if t2 > self._t_ref:
            print(f"\nprogressing time to {t2}")
            for i, ball in enumerate(self.balls):
                print(i, ball.r, ball.v)
        gc_heap = self._constructGlobalBallsCollisionHeap(t2)
        gnw_heap = self._constructGlobalWallCollisionHeap(t2)
        cnt = 0
        while self._progressToNextCollision(t2, gc_heap, gnw_heap):
            cnt += 1
        self.t = t2
        self._updateBallsTime()
        if check_overlap:
            msg = self.anyOverlapMessage()
            if msg:
                print(msg)
                print(f"Count = {cnt}")
        
        return cnt
    
    def calculateTotalKineticEnergy(self) -> Real:
        """
        Calculates the current total translational kinetic energy of
        the objects in the simulation.
        
        Returns:
        Real numeric value giving the total translational kinetic
        energy of the objects in the simulation at the time represented
        by attribute t in terms of the simulation's energy units.
        """
        return sum(ball.calculateKineticEnergy()\
                for ball in self.balls)
    
    def calculateTotalPotentialEnergy(self) -> Real:
        """
        Calculates the current total gravitational potential energy of
        the objects in the simulation.
        The gravitational potential energy is measured relative to the
        spatial origin of the simulation (i.e. if a point mass is at
        the spatial origin of the simulation, then its gravitational
        potential energy is defined to be zero).
        
        Returns:
        Real numeric value giving the total gravitational potential
        energy of the objects in the simulation at the time represented
        by attribute t in terms of the simulation's energy units.
        """
        return sum(ball.calculatePotentialEnergy()\
                for ball in self.balls)
    
    def calculateTotalMechanicalEnergy(self) -> Real:
        """
        Calculates the current total mechanical energy (the
        translational kinetic energy plus the gravitational potential
        energy) of the objects in the simulation.
        The gravitational potential energy is measured relative to the
        spatial origin of the simulation (i.e. if a point mass is at
        the spatial origin of the simulation, then its gravitational
        potential energy is defined to be zero).
        
        Returns:
        Real numeric value giving the total mechanical energy of the
        objects in the simulation at the time represented by attribute
        t in terms of the simulation's energy units.
        """
        return (self.calculateTotalKineticEnergy() +\
                self.calculateTotalPotentialEnergy())
    
    def detectAnyBallOutsideBox(self, balls_t_updated: bool=False)\
            -> Tuple[Real]:
        """
        Checks whether any Ball objects in the simulation are not
        completely contained within the box. If so, returns details
        of the first Ball object identified such that this is the case.
        
        Args:
            Optional named:
            balls_t_updated (bool): If True, it is assumed that all of
                    the Ball objects in the attribute balls have the
                    same attribute t as that of the simulation (i.e.
                    have the same current time). Otherwise, this is not
                    assumed, and the attribute t of those Ball objects
                    is progressed to match that of the simulation.
                Default: False
        
        Returns:
        Tuple, which:
        - If a Ball object in the simulation has been identified as not
           being entirely within the box, is a 5-tuple which for one
           such Ball object and one of the walls it extends beyond:
           - Index 0 contains a non-negative integer representing the
              index of that Ball object in the attribute balls.
           - Index 1 contains a non-negative integer representing the
              index of the basis vector of the simulation which is
              normal to the wall.
           - Index 2 contains the integer 0 or 1, to specify to which
              of the two walls normal to the basis vector identified in
              Index 1 is being referred, with 0 representing that the
              wall in question is that from which the basis vector
              points into the box (referred to as the near wall), and 1
              representing that the wall in question is that from which
              the basis vector points out of the box (referred to as
              the far wall).
           - Index 3 contains a strictly positive real numeric value
              representing the radius of the Ball object in terms of
              the simulation's distance units.
           - Index 4 contains a real numeric value representing the
              displacement along the direction normal to the wall
              (i.e. along the basis vector identified in index 1)
              from the centre of the Ball object to the wall in terms
              of the simulation's distance units.
              A positive value signifies that the centre of the Ball
              object is inside the box and a negative value that the
              centre is outside the box.
        - Otherwise is an empty tuple.
        """
        if not balls_t_updated:
            self._updateBallsTime()
        for idx, ball in enumerate(self.balls):
            ans = ball.isOutsideBox()
            if not ans: continue
            axis_idx, end_idx = ans
            d = self.box_dims[axis_idx] - ball.r[axis_idx] if axis_idx\
                    else ball.r[axis_idx]
            return idx, axis_idx, end_idx, ball.radius, d
        return ()
    
    def _detectBallsOverlap(self, ball: Ball, start_idx: int=0)\
            -> Tuple[Real]:
        """
        For a given Ball object, checks whether it overlaps with any
        Ball object in the attribute balls for which its index in
        that attribute is no less than start_idx. If such an overlap
        is found, then immediately returns details of the overlap.
        
        It is assumed that the value of attribute t (representing
        the current time in terms of the simulation's time measure)
        is the same for the simulation, the Ball object ball and every
        Ball object in the attribute balls being checked for overlap
        with ball.
        
        Args:
            Required positional:
            ball (Ball object): The Ball object for which overlap with
                    the Ball objects in the attribute balls is being
                    checked.
            
            Optional named:
            start_idx (int): Non-negative integer for which precisely
                    those Ball objects in the attribute balls whose
                    index in this attribute is no less than this
                    integer are checked for overlap with ball.
                Default: 0
        
        Returns:
        Tuple, which:
        - If the Ball object ball is found to overlap with any of the
           checked elements of the attribute balls, is a 3-tuple which
           for one such element (referred to as the other Ball object):
           - Index 0 contains a non-negative integers representing
              the index the other Ball object in the attribute balls.
           - Index 1 contains a strictly positive real numeric value
              representing the sum of the radii of the Ball object ball
              and the other Ball object (in terms of the simulation's
              distance units).
           - Index 2 contains a non-negative real numeric value
              representing the square of the current distance between
              the centre of the Ball object ball and the other Ball
              object (in terms of the simulation's distance units
              squared).
        - Otherwise is an empty tuple.
        """
        # Assumes ball and all elements of self.balls have time equal
        # to self.t
        for idx in range(start_idx, len(self.balls)):
            ball2 = self.balls[idx]
            r1, r2 = ball.r, ball2.r
            rad_sum = ball.radius + ball2.radius
            rad_sum_sq = rad_sum ** 2
            d_sq = sum((x - y) ** 2 for x, y in zip(r1, r2))
            if d_sq < rad_sum_sq:
                return idx, rad_sum, d_sq
        return ()
    
    def detectAnyBallsOverlap(self, balls_t_updated: bool=False)\
            -> Tuple[Real]:
        """
        Examines all pairs of objects in the simulation until one pair
        of objects are found to overlap with each other (i.e. the
        intersection of the sets position vectors comprising the two
        objects is not empty). If such an overlap is found, then
        immediately returns details of the overlap.
        During the simulation, no pair of objects should ever overlap
        with each other. Thus, this method acts as a check for errors
        in the correct addition of objects into the simulation and
        for identifying when collision detection and/or resolution for
        collisions between two objects in the simulation is not working
        correctly.
        
        Args:
            Optional named:
            balls_t_updated (bool): If True, it is assumed that all of
                    the Ball objects in the attribute balls have the
                    same attribute t as that of the simulation (i.e.
                    have the same current time). Otherwise, this is not
                    assumed, and the attribute t of those Ball objects
                    is progressed to match that of the simulation.
                Default: False
        
        Returns:
        Tuple, which:
        - If a pair of Ball objects in the simulations have been found
           to overlap with each other, is a 4-tuple which for one such
           pair of Ball objects:
           - Indices 0 and 1 contain non-negative integers representing
              the indices of the two Ball objects in the attribute
              balls (in ascending order).
           - Index 2 contains a strictly positive real numeric value
              representing the sum of the radii of the two Ball objects
              (in terms of the simulation's distance units).
           - Index 3 contains a non-negative real numeric value
              representing the square of the current distance between
              the centres of the two Ball objects (in terms of the
              simulation's distance units squared).
        - Otherwise is an empty tuple.
        """
        if not balls_t_updated:
            self._updateBallsTime()
        for idx1, ball1 in enumerate(self.balls):
            ans = self._detectBallsOverlap(ball1, start_idx=idx1 + 1)
            if ans:
                idx2, rad_sum, d_sq = ans
                return idx1, idx2, rad_sum, d_sq
        return ()
    
    def anyOverlapMessage(self, balls_t_updated: bool=False) -> str:
        """
        Examines every object in the simulation to check whether at
        the current time of the simulation any object is not completely
        within the box, and all pairs of objects in the simulation to
        check for overlap. If at least one of these is the case,
        returns a message giving details of one of these occurrences.
        During the simulation, neither of the above detailed situations
        should occur. Thus, this method acts as a check for errors
        in the correct addition of objects into the simulation and
        for collision detection and/or resolution for both collisions
        between two objects in the simulation and collisions between
        objects and walls.
        
        Args:
            Optional named:
            balls_t_updated (bool): If True, it is assumed that all of
                    the Ball objects in the attribute balls have the
                    same attribute t as that of the simulation (i.e.
                    have the same current time). Otherwise, this is not
                    assumed, and the attribute t of those Ball objects
                    is progressed to match that of the simulation.
                Default: False
        
        Returns:
        A string (str). If at the current time any of its objects
        is not completely within the box or there exists a pair of
        objects in the simulation which overlap with each other, then
        the string contains details of one of these occurrences.
        Otherwise the string is empty.
        """
        if not balls_t_updated:
            self._updateBallsTime()
        ans1 = self.detectAnyBallOutsideBox(balls_t_updated=True)
        t = self.t
        if ans1:
            idx, axis_idx, end_idx, radius, dist = ans1
            end_str = "far" if end_idx else "near" 
            return "At least one Ball object detected at least "\
                    f"partly outside the box at simulation time {t} "\
                    "(in terms of the simulation's time measure) "\
                    f"with the Ball object with index {idx} a "\
                    f"distance {dist} from the {end_str} wall normal "\
                    f"to the axis {axis_idx}, which is less than the "\
                    f"radius of the Ball object, {radius} (with "\
                    "distances given in terms of the simulation's "\
                    "distance units and a negative distance to the "\
                    "wall signifying that the centre of the Ball "\
                    "object has itself gone beyond the wall."
        ans2 = self.detectAnyBallsOverlap(balls_t_updated=True)
        if ans2:
            idx1, idx2, rad_sum, d_sq = ans2
            d = math.sqrt(d_sq)
            return f"Overlap detected at time {t} (in terms of the "\
                    "simulation's time measure) between at least "\
                    f"one pair of Ball objects, including those with "\
                    f"indices {idx1} and {idx2} whose centres are "\
                    f"separated by a distance {d}, which is less "\
                    f"than the sum of their radii {rad_sum} (both "\
                    "in terms of the simulation's distance units)."
        return ""
