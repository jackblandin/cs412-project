from tabulate import tabulate

from envs.multiagent_particle_envs.multiagent.policy import Policy


class VerySimpleSpreadPolicy(Policy):
    """Base class for policies in the VerySimpleSpread Scenario."""

    def __init__(self):
        raise NotImplementedError()

    def action(self, obs):
        """Returns the best action, given an observation.

        Parameters
        ----------
        obs : list
            <'p_velx', 'p_vely', 'p_posx', 'p_posy', 'landmark1_posx',
            'landmark1_posy', 'landmark2_posx', 'landmark2_posy', 'ag_posx',
            'ag_posy', 'ag_com1', 'ag_com2'>

        Returns
        -------
        list<float>, length 6
            Each action index is a float between 0 and 1.

        NOTE
        ----
        Action indexes are:
            0: No-op
            1: Move left
            2: Move right
            3: Move up
            4: Move down
            5: Communicate dimension 1
            6: Communicate dimension 2
        """
        raise NotImplementedError()

    def _obs_str(self, obs):
        """Returns printable English description of observation. Used for
        debugging.

        Parameters
        ----------
        obs : list
            <'p_velx', 'p_vely', 'p_posx', 'p_posy', 'lm1_posx', 'lm_posy',
            'lm2_posx', 'lm2_posy', 'a1_posx', 'a1_posy', 'a1_com1',
            'a1_com2'>

        Returns
        -------
        str
            English description of observation.
        """
        obs = [round(o, 3) for o in obs]
        headers = ['vel_x', 'vel_y', 'pos_x', 'pos_y', 'lm1_x', 'lm1_y',
                   'lm2_x', 'lm2_y', 'a1_x', 'a1_x', 'a1_com1', 'a1_com2']
        return tabulate([obs], headers=headers)

    def _action_str(self, action):
        """Returns printable English description of action. Used for debugging.

        Parameters
        ----------
        action : list, length 6
            Action.

        Returns
        -------
        str
            English description of action.
        """
        action = [round(a, 3) for a in action]
        headers = ['No-op', 'Move left', 'Move right', 'Move up', 'Move down',
                   'Communicate 1', 'Communicate 2']
        return tabulate([action], headers=headers)
