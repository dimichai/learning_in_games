import numpy as np
from dataclasses import dataclass


@dataclass
class GameConfig:
    """
    The minimal parameters to specify games.
    This excludes the payoff functions which are defined separately as functions.
    """
    n_agents: int
    n_actions: int
    n_states: int
    n_iter: int


@dataclass
class RouteConfig(GameConfig):
    cost: float
    
@dataclass
class NSourcesGameConfig:
    n_sources: int
    n_agents_by_source: list
    n_states_by_source: list
    n_actions_by_source: list
    n_iter: int
    
    @property
    def n_agents(self):
        return sum(self.n_agents_by_source)
    
    

def braess_initial_network(actions_1, actions_2, config: NSourcesGameConfig):
    """
    Network from the Braess Paradox without the added link, and the Nash Equilibrium average travel time is 1.00015,
    which is also the optimal average travel time is 1.00015 where players split evenly over the two paths (0.5, 0.5).
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()
    
    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()

    r_up = 1.0001 + (n1_up + n2_up) / n_agents
    r_down = 1.0001 + (n1_down + n2_down) / n_agents
    
    T1 = np.array([-r_up, -r_down])
    T2 = np.array([-r_up, -r_down])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)

    return R, T

def braess_augmented_network(actions_1, actions_2, config: NSourcesGameConfig):
    """
    Network from the Braess Paradox with the added link, and the Nash Equilibrium average travel time is 2,
    but the optimal average travel time is 1.00015: and no players take the added link.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()
    n1_cross = (actions_1 == 2).sum()
    
    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()
    n2_cross = (actions_2 == 2).sum()

    r_up = 1.0001 + (n1_up + n1_cross + n2_up + n2_cross) / n_agents
    r_down = 1.0001 + (n1_down + n1_cross + n2_down + n2_cross) / n_agents
    r_cross = (n1_up + n1_cross + n2_up + n2_cross) / n_agents + (n1_down + n1_cross + n2_down + n2_cross) / n_agents

    T1 = np.array([-r_up, -r_down, -r_cross])
    T2 = np.array([-r_up, -r_down, -r_cross])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T

def fairness_braess_simple(actions_1, actions_2, config: NSourcesGameConfig):
    """
    Network from the Braess Paradox with two sources and one destination.
    """

    n1_up = (actions_1 == 0).sum()
    n1_cross = (actions_1 == 1).sum()

    n2_down = (actions_2 == 0).sum()

    r1_up = 2.0001 + n1_up / config.n_agents_1
    r1_cross = (n1_up + n1_cross) / config.n_agents_1 + (n2_down + n1_cross) / config.n_agents_2
    
    r2_down = 1.0001 + (n2_down + n1_cross) / config.n_agents_2

    T1 = np.array([-r1_up, -r1_cross])
    T2 = np.array([-r2_down])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T

def fairness_braess(actions_1, actions_2, config: NSourcesGameConfig, cost_A2D=1.0001):
    """
    Network from the Braess Paradox with two sources and one destination.
    """

    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()

    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()
    
    cap_A1C = config.n_agents_by_source[0]
    cap_A2C = config.n_agents_by_source[1]
    # cap_DB = config.n_agents_by_source[1]
    cap_DB = config.n_agents
    
    flow_A1C = n1_up
    flow_A2C = n2_up
    flow_DB = n1_down + n2_down

    r1_up = 1.0001 + flow_A1C / cap_A1C
    r1_down = 1.0001 + flow_DB / cap_DB
    
    r2_up = 1.0001 + flow_A2C / cap_A2C
    r2_down = cost_A2D + flow_DB / cap_DB

    T1 = np.array([-r1_up, -r1_down])
    T2 = np.array([-r2_up, -r2_down])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T


def fairness_braess_intervention_1(actions_1, actions_2, config: NSourcesGameConfig, cost_A2D=1.0001):
    """
    Network from the Braess Paradox with two sources and one destination.
    """
    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()
    n1_down_up = (actions_1 == 2).sum()
    n1_down_down = (actions_1 == 3).sum()

    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()
    
    cap_A1C = config.n_agents_by_source[0]
    cap_A2C = config.n_agents_by_source[1]
    # cap_DB =  config.n_agents_by_source[1]
    cap_DB = config.n_agents
    
    flow_A1C = n1_up
    flow_A2C = n2_up + n1_down_up
    flow_DB = n1_down + n2_down + n1_down_down

    r1_up = 1.0001 + flow_A1C / cap_A1C
    r1_down = 1.0001 + flow_DB / cap_DB
    r1_down_up = 1.0001 + flow_A2C/ cap_A2C
    r1_down_down = cost_A2D + flow_DB / cap_DB
    
    r2_up = flow_A2C / cap_A2C + 1.0001
    r2_down = cost_A2D + flow_DB / cap_DB

    T1 = np.array([-r1_up, -r1_down, -r1_down_up, -r1_down_down])
    T2 = np.array([-r2_up, -r2_down])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T

def fairness_braess_intervention_2(actions_1, actions_2, config: NSourcesGameConfig, cost_A2D=1.0001):
    """
    Network from the Braess Paradox with two sources and one destination.
    """
    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()
    n1_upshortcut = (actions_1 == 2).sum()

    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()
    n2_upshortcut = (actions_2 == 2).sum()
    
    cap_A1C = config.n_agents_by_source[0]
    cap_A2C = config.n_agents_by_source[1]
    # cap_DB =  config.n_agents_by_source[1]
    cap_DB = config.n_agents
    
    flow_A1C = n1_up + n1_upshortcut
    flow_A2C = n2_up + n2_upshortcut
    flow_DB = n1_down + n2_down + n1_upshortcut + n2_upshortcut
    
    r1_up = 1.0001 + flow_A1C / cap_A1C
    r1_down = 1.0001 + flow_DB/ cap_DB
    r1_upshortcut = flow_A1C / cap_A1C + flow_DB / cap_DB
    
    r2_up = flow_A2C / cap_A2C + 1.0001
    r2_down = cost_A2D + flow_DB / cap_DB
    r2_upshortcut = flow_A2C / cap_A2C + flow_DB / cap_DB

    T1 = np.array([-r1_up, -r1_down, -r1_upshortcut])
    T2 = np.array([-r2_up, -r2_down, -r2_upshortcut])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T

def fairness_braess_interventions_1_and_2(actions_1, actions_2, config: NSourcesGameConfig, cost_A2D=1.0001):
    """
    Network from the Braess Paradox with two sources and one destination.
    """
    n1_up = (actions_1 == 0).sum()
    n1_down = (actions_1 == 1).sum()
    n1_up_shortcut = (actions_1 == 2).sum()
    n1_down_up = (actions_1 == 3).sum()
    n1_down_down = (actions_1 == 4).sum()
    n1_down_shortcut = (actions_1 == 5).sum()

    n2_up = (actions_2 == 0).sum()
    n2_down = (actions_2 == 1).sum()
    n2_up_shortcut = (actions_2 == 2).sum()
    
    cap_A1C = config.n_agents_by_source[0]
    cap_A2C = config.n_agents_by_source[1]
    # cap_DB =  config.n_agents_by_source[1]
    cap_DB = config.n_agents
    
    flow_A1C = n1_up + n1_up_shortcut
    flow_A2C = n1_down_up + n1_down_shortcut + n2_up + n2_up_shortcut
    flow_DB = n1_down + n1_up_shortcut + n1_down_down + n1_down_shortcut + n2_down + n2_up_shortcut

    r1_up = 1.0001 + flow_A1C / cap_A1C
    r1_down = 1.0001 + flow_DB / cap_DB
    r1_up_shortcut = flow_A1C / cap_A1C + flow_DB / cap_DB
    r1_down_up = 1.0001 + flow_A2C / cap_A2C
    r1_down_down = cost_A2D + flow_DB / cap_DB
    r1_down_shortcut = 1.0001 + flow_A2C / cap_A2C + flow_DB / cap_DB
    
    r2_up = flow_A2C / cap_A2C + 1.0001
    r2_down = cost_A2D + flow_DB / cap_DB
    r2_up_shortcut = flow_A2C / cap_A2C + flow_DB / cap_DB

    T1 = np.array([-r1_up, -r1_down, -r1_up_shortcut, -r1_down_up, -r1_down_down, -r1_down_shortcut])
    T2 = np.array([-r2_up, -r2_down, -r2_up_shortcut])
    R = (T1[actions_1], T2[actions_2])
    T = (T1, T2)
    
    return R, T


def amsterdam_metro(actions_w, actions_e, config: NSourcesGameConfig, crowding_multiplier=1.0, intervention_north_south=False, intervention_west_amstel=False):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    """
    
    # w: west, s: south, e: east, c: central, a: amstel
    west_south_amstel_central = (actions_w == 0).sum()
    if intervention_north_south:
        west_south_central = (actions_w == 1).sum()
    else:
        west_south_central = 0
        
    if intervention_west_amstel:
        west_amstel_central = (actions_w == 2).sum()
    else:
        west_amstel_central = 0
    
    east_amstel_central = (actions_e == 0).sum()
    east_south_amstel_central = (actions_e == 1).sum()
    if intervention_north_south:
        east_south_central = (actions_e == 2).sum()
    else:
        east_south_central = 0
        
    # cap_west_south = 1
    # cap_east_amstel = 1
    # cap_amstel_central = 1
    
    # cap_west_south = 2
    # cap_east_amstel = 1
    # cap_amstel_central = 2
    
    # cap_west_south = 2
    # cap_east_amstel = 2
    # cap_amstel_central = 3
    
    # cap_west_south = 2
    # cap_east_amstel = 1
    # cap_amstel_central = 3

    # cap_south_amstel = 1
    # cap_east_south = 1
    # cap_south_central = 1 if intervention else 0
    
    cap_west_south = config.n_agents
    cap_east_amstel = config.n_agents
    cap_amstel_central = config.n_agents

    cap_south_amstel = config.n_agents
    cap_east_south = config.n_agents
    cap_south_central = config.n_agents if intervention_north_south else 0
    cap_west_amstel = config.n_agents if intervention_west_amstel else 0
    
    flow_west_south = west_south_amstel_central + west_south_central
    flow_south_amstel = west_south_amstel_central + east_south_amstel_central
    flow_east_south = east_south_central + east_south_amstel_central
    flow_east_amstel = east_amstel_central
    flow_amstel_central = west_south_amstel_central + east_amstel_central + east_south_amstel_central + west_amstel_central
    if intervention_north_south:
        flow_south_central = west_south_central + east_south_central
    if intervention_west_amstel:
        flow_west_amstel = west_amstel_central
        
    rw_west_south_amstel_central = 1.7001 + flow_west_south / cap_west_south + 1.1001 + flow_south_amstel / cap_south_amstel + 1.001 + flow_amstel_central / cap_amstel_central
    rw_east_amstel_central = 1.0001 + flow_east_amstel / cap_east_amstel + 1.0001 + flow_amstel_central / cap_amstel_central
    rw_east_south_amstel_central = 1.5001 + flow_east_south / cap_east_south + 1.1001 + flow_south_amstel / cap_south_amstel + 1.001 + flow_amstel_central / cap_amstel_central
    
    if intervention_north_south:
        rw_west_south_central = 1.7001 + flow_west_south / cap_west_south + 1.2001 + flow_south_central /cap_south_central
        rw_east_south_central = 1.5001 + flow_east_south / cap_east_south + 1.2001 + flow_south_central / cap_south_central
        
    if intervention_west_amstel:
        rw_west_amstel_central = 1.4001 + flow_west_amstel / cap_west_amstel + 1.0001 + flow_amstel_central / cap_amstel_central
    
    # rw_west_south_amstel_central = (1 + 1.7) * (flow_west_south / cap_west_south) * crowding_multiplier + (1 + 1.1) * (flow_south_amstel / cap_south_amstel) * crowding_multiplier + (1 + 1.0) * (flow_amstel_central / cap_amstel_central) * crowding_multiplier
    # rw_east_amstel_central = (1 + 1.0) * (flow_east_amstel / cap_east_amstel) * crowding_multiplier + (1 + 1.0) * (flow_amstel_central / cap_amstel_central) * crowding_multiplier
    # rw_east_south_amstel_central = (1 + 1.5) * (flow_east_south / cap_east_south) * crowding_multiplier + (1 + 1.1) * (flow_south_amstel / cap_south_amstel) * crowding_multiplier + (1 + 1.0) * (flow_amstel_central / cap_amstel_central) * crowding_multiplier
    
    # if intervention_north_south:
    #     rw_west_south_central = (1 + 1.7) * (flow_west_south / cap_west_south) * crowding_multiplier + (1 + 1.2) * (flow_south_central / cap_south_central) * crowding_multiplier
    #     rw_east_south_central = (1 + 1.5) * (flow_east_south / cap_east_south) * crowding_multiplier + (1 + 1.2) * (flow_south_central / cap_south_central) * crowding_multiplier
        
    # if intervention_west_amstel:
    #     rw_west_amstel_central = (1 + 1.4) * (flow_west_amstel / cap_west_amstel) * crowding_multiplier + (1 + 1.0) * (flow_amstel_central / cap_amstel_central) * crowding_multiplier
        
    
    if not intervention_north_south and not intervention_west_amstel:
        # Neither intervention
        T1 = np.array([-rw_west_south_amstel_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central])

    elif intervention_north_south and not intervention_west_amstel:
        # Only intervention_north_south
        T1 = np.array([-rw_west_south_amstel_central, -rw_west_south_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central, -rw_east_south_central])

    elif intervention_north_south and intervention_west_amstel:
        # Both interventions
        T1 = np.array([-rw_west_south_amstel_central, -rw_west_south_central, -rw_west_amstel_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central, -rw_east_south_central])

    else:
        # Only intervention_west_amstel is True, which is invalid
        raise ValueError("Invalid combination of interventions: 'intervention_west_amstel' cannot be True without 'intervention_north_south'")
    
    R = (T1[actions_w], T2[actions_e])
    T = (T1, T2)
    
    return R, T

def amsterdam_metro_north_south_intervention(actions_w, actions_e, config: NSourcesGameConfig):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    This is a variant of the amsterdam_metro function where the intervention is applied.
    """
    return amsterdam_metro(actions_w, actions_e, config, intervention_north_south=True)

def amsterdam_metro_north_south_west_amstel_intervention(actions_w, actions_e, config: NSourcesGameConfig):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    This is a variant of the amsterdam_metro function where the intervention is applied.
    """
    return amsterdam_metro(actions_w, actions_e, config, intervention_north_south=True, intervention_west_amstel=True)

def amsterdam_metro_with_south(actions_w, actions_e, actions_z, config: NSourcesGameConfig, intervention=False):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    """
    
    # w: west, s: south, e: east, c: central, a: amstel
    west_south_amstel_central = (actions_w == 0).sum()
    if intervention:
        west_south_central = (actions_w == 1).sum()
    else:
        west_south_central = 0
    
    east_amstel_central = (actions_e == 0).sum()
    east_south_amstel_central = (actions_e == 1).sum()
    if intervention:
        east_south_central = (actions_e == 2).sum()
    else:
        east_south_central = 0
        
        
    south_amstel_central = (actions_z == 0).sum()
    if intervention:
        south_central = (actions_z == 1).sum()
        
    cap_west_south = 2
    cap_east_amstel = 1
    cap_amstel_central = 3

    cap_south_amstel = 1
    cap_east_south = 1
    cap_south_central = 1 if intervention else 0
    
    flow_west_south = west_south_amstel_central + west_south_central
    flow_south_amstel = west_south_amstel_central + east_south_amstel_central + south_amstel_central
    flow_east_south = east_south_central + east_south_amstel_central
    flow_east_amstel = east_amstel_central
    flow_amstel_central = west_south_amstel_central + east_amstel_central + east_south_amstel_central + south_amstel_central
    if intervention:
        flow_south_central = west_south_central + east_south_central + south_central
        
    rw_west_south_amstel_central = flow_west_south / cap_west_south + flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    rw_east_amstel_central = flow_east_amstel / cap_east_amstel + flow_amstel_central / cap_amstel_central
    rw_east_south_amstel_central = flow_east_south / cap_east_south + flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    rw_south_amstel_central = flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    
    if intervention:
        rw_west_south_central = flow_west_south / cap_west_south + flow_south_central /cap_south_central
        rw_east_south_central = flow_east_south / cap_east_south + flow_south_central / cap_south_central
        rw_south_central = flow_south_central / cap_south_central
    
    if intervention:
        T1 = np.array([-rw_west_south_amstel_central, -rw_west_south_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central, -rw_east_south_central])
        T3 = np.array([-rw_south_amstel_central, -rw_south_central])
    else:
        T1 = np.array([-rw_west_south_amstel_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central])
        T3 = np.array([-rw_south_amstel_central])
    
    R = (T1[actions_w], T2[actions_e], T3[actions_z])
    T = (T1, T2, T3)
    
    return R, T

def amsterdam_metro_with_south_north_south_intervention(actions_w, actions_e, actions_z, config: NSourcesGameConfig):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    This is a variant of the amsterdam_metro function where the intervention is applied.
    """
    return amsterdam_metro_with_south(actions_w, actions_e, actions_z, config, intervention=True)



def amsterdam_rail(actions_z, actions_e, config: NSourcesGameConfig, intervention=False):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    """
    
    # z: zuidas, w: west, s: south, e: east, c: central, a: amstel
    zuidas_south_west_central = (actions_z == 0).sum()
    zuidas_south_amstel_central = (actions_z == 1).sum()
    if intervention:
        zuidas_south_central = (actions_z == 2).sum()
    else:
        zuidas_south_central = 0
    
    east_amstel_central = (actions_e == 0).sum()
    east_south_west_central = (actions_e == 1).sum()
    if intervention:
        east_south_central = (actions_e == 2).sum()
    else:
        east_south_central = 0
    
    cap_zuidas_south = 1 
    cap_south_west = 1 
    cap_west_central = 1
    cap_south_amstel = 1
    cap_amstel_central = 1
    cap_east_amstel = 1
    cap_east_south = 1 
    cap_south_central = 1 if intervention else 0
    
    flow_zuidas_south = zuidas_south_west_central + zuidas_south_amstel_central + zuidas_south_central
    flow_south_west = zuidas_south_west_central + east_south_west_central
    flow_west_central = zuidas_south_west_central + east_south_west_central
    flow_south_amstel = zuidas_south_amstel_central
    flow_amstel_central = zuidas_south_amstel_central + east_amstel_central
    flow_east_amstel = east_amstel_central
    flow_east_south = east_south_west_central + east_south_central
    if intervention:
        flow_south_central = zuidas_south_central + east_south_central
        
    rw_zuidas_south_west_central = 1.0001 + flow_zuidas_south / cap_zuidas_south + flow_south_west / cap_south_west + flow_west_central / cap_west_central
    rw_zuidas_south_amstel_central = 1.0001 + flow_zuidas_south / cap_zuidas_south + flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    rw_east_amstel_central = 1.0001 + flow_east_amstel / cap_east_amstel + flow_amstel_central / cap_amstel_central
    rw_east_south_west_central = 1.0001 + flow_east_south / cap_east_south + flow_south_west / cap_south_west + flow_west_central / cap_west_central
    
    if intervention:
        rw_zuidas_south_central = 1.0001 + flow_zuidas_south / cap_zuidas_south + flow_south_central / cap_south_central
        rw_east_south_central = 1.0001 + flow_east_south / cap_east_south + flow_south_central / cap_south_central
    
    if intervention:
        T1 = np.array([-rw_zuidas_south_west_central, -rw_zuidas_south_amstel_central, -rw_zuidas_south_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_west_central, -rw_east_south_central])
    else:
        T1 = np.array([-rw_zuidas_south_west_central, -rw_zuidas_south_amstel_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_west_central])
    
    R = (T1[actions_z], T2[actions_e])
    T = (T1, T2)
    
    return R, T

def amsterdam_rail_intervention(actions_z, actions_e, config: NSourcesGameConfig):
    return amsterdam_rail(actions_z, actions_e, config, intervention=True)


def amsterdam_v3(actions_w, actions_e, config: NSourcesGameConfig, intervention=False):
    """
    Network from the Amsterdam Metro with two sources and one destination.
    """
    
    west_lely_central = (actions_w == 0).sum()
    west_south_amstel_central = (actions_w == 1).sum()
    if intervention:
        west_south_central = (actions_w == 2).sum()
    else:
        west_south_central = 0
    
    east_amstel_central = (actions_e == 0).sum()
    east_south_amstel_central = (actions_e == 1).sum()
    if intervention:
        east_south_central = (actions_e == 2).sum()
    else:
        east_south_central = 0
    
    cap_zuidas_south = 1
    cap_south_west = 1 
    cap_west_central = 1
    cap_south_amstel = 1
    cap_amstel_central = 1000000
    cap_east_amstel = 1
    cap_east_south = 1
    cap_south_central = 1 if intervention else 0
    cap_west_lely = 1
    cap_lely_central = 1
    cap_west_south = 1
    
    flow_west_lely = west_lely_central
    flow_lely_central = west_lely_central
    flow_west_south = west_south_amstel_central + west_south_central
    flow_south_amstel = west_south_amstel_central + east_south_amstel_central
    flow_amstel_central = west_south_amstel_central + east_amstel_central + east_south_amstel_central
    flow_east_amstel = east_amstel_central
    flow_east_south = east_south_central + east_south_amstel_central
    if intervention:
        flow_south_central = west_south_central + east_south_central
        
    rw_west_lely_central = 1.0001 + flow_west_lely / cap_west_lely + flow_lely_central / cap_lely_central
    rw_west_south_amstel_central = 1.0001 + flow_west_south / cap_west_south + flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    
    rw_east_amstel_central = 1.0001 + flow_east_amstel / cap_east_amstel + flow_amstel_central / cap_amstel_central
    rw_east_south_amstel_central = 1.0001 + flow_east_south / cap_east_south + flow_south_amstel / cap_south_amstel + flow_amstel_central / cap_amstel_central
    
    if intervention:
        rw_west_south_central =  1.0001 + flow_west_south / cap_west_south + flow_south_central / cap_south_central
        rw_east_south_central = 1.0001 + flow_east_south / cap_east_south + flow_south_central / cap_south_central
    
    if intervention:
        T1 = np.array([-rw_west_lely_central, -rw_west_south_amstel_central, -rw_west_south_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central, -rw_east_south_central])
    else:
        T1 = np.array([-rw_west_lely_central, -rw_west_south_amstel_central])
        T2 = np.array([-rw_east_amstel_central, -rw_east_south_amstel_central])
    
    R = (T1[actions_w], T2[actions_e])
    T = (T1, T2)
    
    return R, T

def amsterdam_v3_intervention(actions_w, actions_e, config: NSourcesGameConfig):
    return amsterdam_v3(actions_w, actions_e, config, intervention=True)


def two_route_game(actions, config: RouteConfig):
    """
    A two path routing game where the cost parameter can be used to vary the edge costs from one extreme, where the
    network resembles the Pigou network, to another extreme where the Nash Equilibrium corresponds to the optimal
    average travel time.
    :param actions: np.ndarray of Actions indexed by agents
    :param cost:
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents + config.cost
    r_1 = (1 - n_up / n_agents) + (1 - config.cost)

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou(actions, config):
    """
    The Pigou network routing game which has two paths, one with a fixed cost and the other with a variable cost equal
    to the percentage of players that take that path. The classic Pigou game has a fixed cost of 1.0001
    :param actions: np.ndarray of Actions indexed by agents
    :param cost:
    :return:
    """
    n_agents = config.n_agents
    n_down = (actions == 1).sum()
    pct = n_down / n_agents

    r_0 = config.cost + 0.00001
    r_1 = pct

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou3(actions, config: GameConfig):
    """
    A version of a Pigou network with three paths, two fixed cost paths and one variable cost path.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents
    r_1 = 1
    r_2 = 1

    T = np.array([-r_0, -r_1, -r_2])
    R = T[actions]
    return R, T


@dataclass
class MinorityConfig(GameConfig):
    threshold: float


def minority_game(actions, config: MinorityConfig):
    """
    A minority game where the minority group is determined by a threshold
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_up = (actions == 0).sum()

    if n_agents * config.threshold >= n_up:  # up is minority
        r_0 = 1
        r_1 = 0
    else:
        r_0 = 0
        r_1 = 1

    T = np.array([r_0, r_1])
    R = T[actions]
    return R, T


def minority_game_2(actions, config: GameConfig):
    """
    A minority game variant.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_agents = config.n_agents
    n_a = (actions == 0).sum()
    fraction_a = n_a / n_agents
    fraction_b = 1 - fraction_a

    r_a = 1 - 2 * fraction_a
    r_b = 1 - 2 * fraction_b

    T = np.array([r_a, r_b])
    R = T[actions]
    S = None  # stateless
    return R, S


def el_farol_bar(actions, config: MinorityConfig):
    """
    The El Farol Bar game where all those that stay home get a payoff of 1, while
    all those that go to the bar get payoffs better than 1 only if the fraction (pct)
    of players that go to the bar is below a threshold.
    :param actions: np.ndarray of Actions indexed by agents
    :return:
    """
    n_agents = len(actions)
    n_bar = (actions == 1).sum()
    pct = n_bar / n_agents

    r_0 = 1
    r_1 = 2 - 4 * pct if (pct > config.threshold) else 4 * pct - 2

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def duopoly(actions, config: GameConfig):
    """
    A duopoly pricing game, intended to be played as a turn taking game, but can also be played as a simultaneous game.
    The state of the game for each player is the previous action of the other player.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    a1 = actions[0]
    a2 = actions[1]

    p1 = a1 / config.n_actions
    p2 = a2 / config.n_actions

    if p1 < p2:
        r1 = (1 - p1) * p1
        r2 = 0
    elif p1 == p2:
        r1 = 0.5 * (1 - p1)
        r2 = r1
    elif p1 > p2:
        r1 = 0
        r2 = (1 - p2) * p2

    R = np.array([r1, r2])
    S = np.array([a2, a1])

    return R, S


@dataclass
class PrisonersDilemmaConfig(GameConfig):
    reward_payoff: float
    suckers_payoff: float


def prisoners_dilemma(actions, config: PrisonersDilemmaConfig):
    """
    The Prisoner's Dilemma game parameterized by the reward and suckers payoffs.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    a1 = actions[0]
    a2 = actions[1]

    if a1 == 0 and a2 == 0:
        r1 = config.reward_payoff
        r2 = config.reward_payoff
    elif a1 == 0 and a2 == 1:
        r1 = -config.suckers_payoff
        r2 = 1
    elif a1 == 1 and a2 == 0:
        r1 = 1
        r2 = -config.suckers_payoff
    elif a1 == 1 and a2 == 1:
        r1 = 0
        r2 = 0

    state = a1 + a2

    R = np.array([r1, r2])
    S = np.array([state, state])

    return R, S


@dataclass()
class PopulationConfig(GameConfig):
    V: float
    K: float
    exponent: float
    cost: float


def population_game(actions, config: PopulationConfig):
    """
    A population game as found in the paper 'Catastrophe by Design in Population Games: A Mechanism to Destabilize
    Inefficient Locked-in Technologies' (https://doi.org/10.1145/3583782).
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    n_players = len(actions)
    fraction_weak = (actions == 0).sum() / n_players
    fraction_strong = (actions == 1).sum() / n_players

    utility_weak = config.V * (fraction_weak * config.K) ** (config.exponent - 1) - config.cost
    utility_strong = config.V * (fraction_strong * config.K) ** (config.exponent - 1)  # no added cost

    T = [utility_weak, utility_strong]
    R = np.array([T[a] for a in actions])
    return R, T


@dataclass
class PublicGoodsConfig(GameConfig):
    multiplier: float
    beta: float


def public_goods_game(actions, config: PublicGoodsConfig):
    """
    A public goods game parametrized by the multiplier, and Beta, a parameter which controls the slope of
    the marginal contributions of each action.
    :param actions: np.ndarray of Actions indexed by agents
    :param config: dataclass of parameters for the game
    :return:
    """
    norm_A = actions / config.n_actions
    pot = config.multiplier * np.power(norm_A, config.beta).sum()
    R = 1 - norm_A + pot
    return R
