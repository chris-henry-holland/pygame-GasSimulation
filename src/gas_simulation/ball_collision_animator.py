#!/usr/bin/env python3
from typing import (
    Union,
    Tuple,
    Set,
    Optional,
)

from collections import deque

import pygame

from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    K_RETURN,
    K_KP_ENTER,
)

from .utils import Real

from .ball_collision_simulator import (
    Ball,
    MultiBallSimulation,
)
        
enter_keys = {K_RETURN, K_KP_ENTER}

named_colors_def = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "light_grey": (150, 150, 150),
    "dark_grey": (100, 100, 100),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}

class Ball2DSprite(pygame.sprite.Sprite):
    """
    A pygame circle sprite representing a 2-dimensional Ball object in
    a MultiBallSimulation so that the Ball object can be displayed on
    screen as part of the animation of the simulation.
    
    Initialisation args:
        
        Required positional:
        
        animation (MultiBallSimulationAnimatorDisplay object): Sets
                the attribute animation, specifying the animation of
                the simulation to which the sprite belongs.
        ball (Ball object): Sets the attribute ball, specifying the
                2-dimensional Ball object the sprite represents. This
                Ball object should be part of the simulation.
        
        Optional named:
        
        color (3-tuple of ints between 0 and 255 inclusive): Sets the
                attribute color, specifying the RGB code of the color
                of the sprite.
            Default: (255, 0, 0)- representing red
    
    Attributes:
        
        Fixed (i.e. set at instance creation and unchanged through the
            lifetime of the object instance):
        
        animation (MultiBallSimulationAnimatorDisplay object): The
                animation of the simulation to which the sprite
                belongs.
        ball (Ball object): The simulation's 2-dimensional Ball object
                the sprite represents.
        n_dims (strictly positive int): The number of spatial
                dimensions of the simulation (must be 2)
        dist_unit (strictly positive real numeric value): 
        radius (strictly positive real numeric value):
        radius_pixel (strictly positive int):
        
        
        Updateable (i.e. can be updated by the user at any point
            during the lifetime of the instance):
        color (3-tuple of ints between 0 and 255 inclusive): RGB color
                code of the sprite.
        
        Derived (i.e. set at instance creation and updated based on
            calculations made using other attribute values):
        r (2-tuple of real numeric values):
        r_pixel (2-tuple of ints):
        surf ():
        rect ():
    
    Methods:
        (For full description, see documentation of the method itself)
        
        draw(): Draws the sprite on the screen.
    """
    
    def __init__(self,\
            animation: "MultiBallSimulationAnimatorDisplay",\
            ball: Ball, color: Tuple[int]=named_colors_def["red"]):
        super(Ball2DSprite, self).__init__()
        
        self._animation = animation
        self._ball = ball
        
        self._dist_unit = animation.dist_unit
        self._radius = ball.radius
        self._radius_pixel = round(self._dist_unit * self._radius)
        
        self._color = color
    
    @property
    def animation(self):
        return self._animation
    
    @property
    def screen(self):
        return self._animation.screen
    
    @property
    def ball(self):
        return self._ball
    
    @property
    def dist_unit(self):
        return self._dist_unit
    
    @property
    def n_dims(self):
        return self.ball.n_dims
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def radius_pixel(self):
        return self._radius_pixel
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, color: Tuple[int]):
        if getattr(self, "_color", None) == color:
            return
        self._color = color
        self._surf = None
        return
    
    @property
    def surf(self):
        if getattr(self, "_surf", None) is None:
            self._updateSurface()
        return self._surf
    
    @property
    def r(self):
        prev = getattr(self, "_r", None)
        res = self.ball.r
        if res != prev:
            self._r = res
            self._r_pixel = None
        return res
    
    @property
    def r_pixel(self):
        r = self.r
        res = getattr(self, "_r_pixel", None)
        if res is None:
            res = self.animation.arenaPixelPosition(r)
            self._r_pixel = res
            self._rect = None
        return res
    
    @property
    def rect(self):
        r_pixel = self.r_pixel
        res = getattr(self, "_rect", None)
        if res is None:
            res = self.surf.get_rect(center=r_pixel)
            self._rect = res
        return res
    
    def _updateSurface(self) -> None:
        """
        Updates the attribute surf, which specifies sprite's
        appearance.
        
        Intended to be used when the sprite is first created and
        after a change of the sprite's properties that would affect
        its appearance (e.g. color). Note that specification of the
        position on the screen is independent of the attribute surf,
        so this does not need to be used after a change that affects
        only the position of the sprite only.
        
        Returns:
        None
        """
        self._surf = pygame.Surface(\
                tuple([self.radius_pixel * 2] * self.n_dims),\
                pygame.SRCALPHA)
        pygame.draw.circle(self._surf, self.color,\
                tuple([self.radius_pixel] * self.n_dims),\
                self.radius_pixel)
        return
    
    def draw(self) -> None:
        """
        Draws the sprite to screen, in accordance with its current
        properties (radius, color and position).
        
        Returns:
        None
        """
        self.screen.blit(self.surf, self.rect)

class MultiBallSimulationAnimatorMain:
    """
    Class whose instances represent the main program for displaying
    on screen simulations of 2-dimensional balls in a rectangular
    box under a uniform gravitational field and perfectly elastic
    collisions (see documentation of MultiBallSimulation for full
    details regarding the parameters of the simualtion).
    
    Initialisation args:
        Optional named:
        dist_unit (strictly positive real numeric value): Sets the
                attribute dist_unit, specifying the number of pixels in
                each of the simulation's distance units.
            Default: 12
        n_unit_area (2-tuple of strictly positive real numeric values):
                Sets the attribute n_unit_area, specifying the
                dimensions of the arena (and so box) in terms of the
                simulation's distance units. Index 0 and 1 specify the
                width and height of the arena respectively.
            Defualt: (16, 15)
        framerate (strictly positive real numeric value): Sets
                the framerate of the animation (i.e. the number of
                times per second the display is refreshed).
            Default: 60
        n_sim_cycle_per_frame (strictly positive int): Sets the number
                cycles of the simulation performed per frame of the
                animation.
            Default: 1
        borders (2-tuple of 2-tuples of strictly positive numeric
                values): Specifies the size of the borders outside
                each edge of the arena, in terms of the simulation's
                distance unit. The border sizes are specified as: 
                 ((left, right), (upper, lower))
            Default: ((1, 1), (4, 1))
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
        
        dist_unit (strictly positive real numeric value):
        arena_dims (
        enter_keys ():
        named_colors ():
        
        Updateable (i.e. can be updated by the user at any point
            during the lifetime of the instance):
        
        
        
        Derived (i.e. set at instance creation and updated based on
            calculations made using other attribute values):
    
    
    Methods:
        (For full description, see documentation of the method itself)
    
    """
    def __init__(self, dist_unit: Real=12,\
            arena_dims: Tuple[Real]=(16, 15),\
            framerate: Real=60,\
            n_sim_cycle_per_frame: Real=1,\
            borders: Tuple[Tuple[Real]]=((1, 1), (4, 1)),\
            g: Union[Real, Tuple[Real]]=0):
        pygame.init()
        
        #self.running = False
        #self.quit = False
        
        self.sim_animator = MultiBallSimulationAnimatorDisplay(\
                main=self, dist_unit=dist_unit,\
                arena_dims=arena_dims,\
                framerate=framerate,\
                n_sim_cycle_per_frame=n_sim_cycle_per_frame,\
                borders=borders, g=g)
        
    @property
    def enter_keys(self):
        return getattr(self, "_enter_keys", enter_keys)
    
    @property
    def named_colors(self):
        return getattr(self, "_named_colors", named_colors_def)
        
    @property
    def borders(self):
        return self.sim_animator.borders
    
    @borders.setter
    def borders(self, borders):
        self.sim_animator.borders = borders
        return
    
    @property
    def dist_unit(self):
        return self.sim_animator.dist_unit
    
    @dist_unit.setter
    def dist_unit(self, dist_unit):
        self.sim_animator.dist_unit = dist_unit
        return
    
    @property
    def arena_dims(self):
        return self.sim_animator.arena_dims
    
    @arena_dims.setter
    def arena_dims(self, arena_dims):
        self.sim_animator.arena_dims = arena_dims
        return
    
    @property
    def screen_dims(self):
        return self.sim_animator.screen_dims
    
    @property
    def screen(self):
        screen = getattr(self, "_screen", None)
        if screen is None:
            self._screen = pygame.display.set_mode(self.screen_dims)
        return self._screen
    
    @property
    def dt_s(self):
        return self.sim_animator.dt_s
    
    def screenPixelPosition(self, pos: Tuple[Real]) -> Tuple[int]:
        return self.sim_animator.screenPixelPosition(pos)
    
    def arenaPixelPosition(self, pos: Tuple[Real]) -> Tuple[int]:
        return self.sim_animator.arenaPixelPosition(pos)
    
    def addBall(self, m: Real, radius: Real,\
            r0: Tuple[Real], v0: Tuple[Real],\
            color: Tuple[int]=named_colors_def["red"],\
            incl_borders: bool=False,\
            balls_t_updated: bool=False,\
            check_overlap: bool=True) -> bool:
        return self.sim_animator.addBall(m, radius, r0, v0,\
                color=color, incl_borders=incl_borders,\
                balls_t_updated=balls_t_updated,\
                check_overlap=check_overlap)
    
    def run(self, print_mechE: bool=False, check_overlap: bool=True)\
            -> None:
        self.run_simulation(print_mechE=print_mechE,\
                check_overlap=check_overlap)
        return
    
    def run_simulation(self, print_mechE: bool=False,\
            check_overlap: bool=True) -> None:
        self.sim_animator.run(print_mechE=print_mechE,\
                check_overlap=check_overlap)
        return

class MultiBallSimulationAnimatorDisplay:
    def __init__(self, main=None, dist_unit=12, arena_dims=(16, 15),
            framerate: int=30,\
            n_sim_cycle_per_frame: int=1,\
            dt_sim_per_sim_cycle: Real=1,\
            borders=((1, 1), (4, 1)), g=0):
        
        self.main = main
        
        self.dist_unit = dist_unit
        self.arena_dims = arena_dims
        self.framerate = framerate
        self.n_sim_cycle_per_frame =\
                n_sim_cycle_per_frame
        self.borders = borders
        
        self.sim_framerate = self.framerate *\
                self.n_sim_cycle_per_frame
        self.dt_sim_per_sim_cycle = dt_sim_per_sim_cycle
        self.dt_sim_per_sec = self.sim_framerate *\
                self.dt_sim_per_sim_cycle
        self.dt_sim_per_sec_sq = self.dt_sim_per_sec ** 2
        
        self.all_sprites = pygame.sprite.Group()
        self.ball_sprites = pygame.sprite.Group()
        
        g_sim = tuple(x / self.dt_sim_per_sec_sq for x in g)\
                if hasattr(g, "__getitem__") else\
                g / self.dt_sim_per_sec_sq
        
        #self.sim = MultiBallSimulation(box_dims=self.arena_dims,\
        #        dt_s=self.dt_s_sim, t0_s=0, g=g_sim)
        self.sim = MultiBallSimulation(box_dims=self.arena_dims,\
                g=g_sim)
    
    @property
    def enter_keys(self):
        return getattr(self.main, "enter_keys", enter_keys)
    
    @property
    def named_colors(self):
        return getattr(self.main, "named_colors", named_colors_def)
    
    @property
    def borders(self):
        return self._borders
    
    @borders.setter
    def borders(self, borders):
        self._arena_ul_pixel = None
        self._borders = borders
    
    @property
    def dist_unit(self):
        return self._dist_unit
    
    @dist_unit.setter
    def dist_unit(self, dist_unit):
        self._arena_dims_pixel = None
        self._screen_dims = None
        self._arena_ul_pixel = None
        self._dist_unit = dist_unit
    
    @property
    def arena_dims(self):
        return self._arena_dims
    
    @arena_dims.setter
    def arena_dims(self, arena_dims):
        self._arena_dims_pixel = None
        self._screen_dims = None
        self._arena_ul_pixel = None
        self._arena_dims = arena_dims
    
    @property
    def arena_dims_pixel(self):
        arena_dims_pixel = getattr(self, "_arena_dims_pixel", None)
        if arena_dims_pixel is not None:
            return arena_dims_pixel
        self._arena_dims_pixel = tuple(self._dist_unit * x for x in\
                                self._arena_dims)
        return self._arena_dims_pixel
    
    @property
    def arena_ul_pixel(self):
        # Position of the upper left corner of the arena in terms of
        # pixels from the upper left corner of the window
        res = getattr(self, "_arena_ul_pixel", None)
        if res is None:
            res = self.arenaPixelPosition(\
                    [0] * len(self.arena_dims))
            self._arena_ul_pixel = res
        return res
    
    @property
    def screen_dims(self):
        screen_dims = getattr(self, "_screen_dims", None)
        if screen_dims is not None:
            return screen_dims
        self._screen_dims = tuple(self.dist_unit * (x + sum(y))\
                for x, y in zip(self.arena_dims, self.borders))
        
        return self._screen_dims
    
    @property
    def screen(self):
        screen = getattr(self, "_screen", None)
        if screen is None:
            if self.main is None:
                self._screen = pygame.display.set_mode(self.screen_dims)
            else:
                self._screen = self.main.screen
        return self._screen
    
    @property
    def arena(self):
        arena = getattr(self, "_arena", None)
        if arena is not None:
            return arena
        self._arena = pygame.draw.rect(self.screen,\
                self.named_colors["white"],\
                [*self.arena_ul_pixel, *self.arena_dims_pixel])
        return self._arena
    
    @property
    def t(self):
        return self.sim.t_s
    
    def screenPixelPosition(self, pos: Tuple[Real]) -> Tuple[int]:
        """
        Based on the dist_unit used and the borders, calculates the actual
        pixel position of a given object on the screen.
        
        Args:
        pos (2-tuple of ints/floats): The position in the units in the
                code (horizontal position, vertical position)
        dist_unit (float/int): How much the units used in the code need to be
                multiplied for the screen
        borders (2-tuple of 2-tuples of floats/ints): the size of the
                borders (in terms of the units used in the code). The order
                is: ((left, right), (upper, lower)).
        
        Returns:
        2-tuple of floats/ints with the actual screen position (horizontal
                position, vertical position).
        """
        return tuple(round(self.dist_unit * x) for x in pos)
    
    def arenaPixelPosition(self, pos: Tuple[Real]) -> Tuple[int]:
        return tuple(round(self.dist_unit * (x + y[0]))\
            for x, y in zip(pos, self.borders))
    
    def checkInputs(self, extra_events: Optional[Set]=None,\
            keys_to_check: Optional[Set]=None):
        
        if extra_events is None:
            extra_events = set()
        extra_seen = set()
        running = True
        quit = False
        for event in pygame.event.get():
            # Check for KEYDOWN event
            if event.type == KEYDOWN:
                # If the Esc key is pressed, then exit the main loop
                if event.key == K_ESCAPE:
                    running = False
                if event.key in extra_events:
                    extra_seen.add(event)
            # Check for QUIT event. If QUIT, then set running to false.
            elif event.type == QUIT:
                quit = True
                running = False
            elif event.type in extra_events:
                extra_seen.add(event)
        if not keys_to_check:
            return (running, quit, extra_seen, set())
        keys_pressed = pygame.key.get_pressed()
        return (running, quit, extra_seen,\
                {x for x in keys_to_check if keys_pressed[x]})
    
    def addBall(self, m: Real, radius: Real,\
            r0: Tuple[Real], v0: Tuple[Real],\
            color: Tuple[int]=named_colors_def["red"],\
            incl_borders: bool=False,\
            balls_t_updated: bool=False,\
            check_overlap: bool=True) -> bool:
        r = r0 if not incl_borders else tuple(x + y[0] for x, y in\
                zip(r0, self.borders))
        v = tuple(x / self.dt_sim_per_sec for x in v0)
        if not self.sim.addBall(m, radius, r, v,\
                balls_t_updated=balls_t_updated,\
                check_overlap=check_overlap):
            return False
        ball = self.sim.balls[-1]
        
        ball_sprite = Ball2DSprite(self, ball, color=color)
        self.all_sprites.add(ball_sprite)
        self.ball_sprites.add(ball_sprite)
        return True
        
    def run(self, print_mechE: bool=False, check_overlap: bool=True)\
            -> bool:
        # Setup the clock for a consistent framerate
        clock = pygame.time.Clock()
        
        input_buffer_qu = deque()
        prev_pressed_keys = None
        
        # Variable to keep the main loop running
        running = True
        
        # Variable signifying whether to completely exit
        quit = False
        
        #print(f"display framerate = {self.framerate}")
        #dt_s = 1 / self.framerate
        #print(f"dt_s = {dt_s}")
        
        # Main loop
        while running:
            (running, quit, prev_pressed_keys) =\
                    self.animationLoop(input_buffer_qu,\
                    prev_pressed_keys, print_mechE=print_mechE,\
                    check_overlap=check_overlap)
            clock.tick(self.framerate)
        return quit
    
    def animationLoop(self, input_buffer_qu, prev_pressed_keys: Optional[Set]=None,\
            print_mechE: bool=False, check_overlap: bool=True)\
            -> Tuple[Union[bool, Optional[set]]]:
        """
        Progresses the simulation forward by one animation frame,
        updating the display to reflect the evolution of the simulation
        and resolving any user inputs.
        
        Args:
            Required positional:
            input_buffer_qu (deque):
            prev_pressed_keys ():
            
            Optional named:
            print_mechE (bool): If True then at the end of each
                    simulation frame prints to console the current
                    total mechanical energy of all of the objects in
                    the simulation. Otherwise, no such console output
                    is printed.
                Default: False
            check_overlap (bool): If True then at the end of each
                    simulation frame inspects each pair of objects
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
        None
        """
        if prev_pressed_keys is None: prev_pressed_keys = set()
        
        # Checking user inputs (keys pressed and mouse clicks)
        (running, quit, extra_seen, pressed_keys) = self.checkInputs()
        if not running:
            return (running, quit, pressed_keys)
            
        # Get the set of keys pressed and check for user input
        if pressed_keys != prev_pressed_keys:
            #key_buffer_list.append(pressed_keys)
            prev_pressed_keys = pressed_keys
        
        for _ in range(self.n_sim_cycle_per_frame):
            self.simCycle(check_overlap=check_overlap)
            if print_mechE:
                print("Total mechanical energy = "\
                        f"{self.calculateTotalMechanicalEnergy()}")
        
        # Fill the borders with grey
        self.screen.fill(self.named_colors["dark_grey"])
        
        # Load the arena
        self._arena = None
        self.arena

        # Draw all sprites
        for sprite in self.all_sprites:
            sprite.draw()
        
        # Update the display
        pygame.display.flip()
        
        return (running, quit, prev_pressed_keys)
    
    def simCycle(self, check_overlap: bool=True) -> None:
        """
        Progresses the simulation forward by one simulation cycle.
        
        Args:
            Optional named:
            check_overlap (bool): If True then at the end of each
                    simulation cycle inspects each pair of objects
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
        None
        """
        cnt = self.sim.progressTime(dt=self.dt_sim_per_sim_cycle,\
                check_overlap=check_overlap)
        return
    
    def calculateTotalKineticEnergy(self) -> Real:
        """
        Calculates the current total kinetic energy of the objects in
        the simulation.
        
        Returns:
        Real numeric value giving the total kinetic energy of the
        objects in the simulation at the time represented by attribute
        t in terms of the animation's energy units.
        """
        return self.sim.calculateTotalKineticEnergy() *\
                self.dt_sim_per_sec_sq
    
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
        by attribute t in terms of the animation's energy units.
        """
        return self.sim.calculateTotalPotentialEnergy() *\
                self.dt_sim_per_sec_sq
    
    def calculateTotalMechanicalEnergy(self) -> Real:
        """
        Calculates the current total mechanical energy (the kinetic
        energy plus the gravitational potential energy) of the objects
        in the simulation.
        The gravitational potential energy is measured relative to the
        spatial origin of the simulation (i.e. if a point mass is at
        the spatial origin of the simulation, then its gravitational
        potential energy is defined to be zero).
        
        Returns:
        Real numeric value giving the total mechanical energy of the
        objects in the simulation at the time represented by attribute
        t in terms of the animation's energy units.
        """
        return self.sim.calculateTotalMechanicalEnergy() *\
                self.dt_sim_per_sec_sq
    
