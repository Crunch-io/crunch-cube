'''Grouping of various cube methods'''

from ..dimension import Dimension


class TableHelper(object):
    '''Groups together useful cube utility methods.'''
    def __init__(self, cube):
        self._cube = cube

    @property
    def all_dimensions(self):
        '''Gets the dimensions of the crunch cube.

        This function is internal, and is not mean to be used by ouside users
        of the CrunchCube class. The main reason for this is the internal
        representation of the different variable types (namely the MR and the
        CA). These types have two dimensions each, but in the case of MR, the
        second dimensions shouldn't be visible to the user. This function
        returns such dimensions as well, since they're necessary for the
        correct implementation of the functionality for the MR type.
        The version that is mentioned to be used by users is the
        property 'dimensions'.
        '''
        entries = self._cube['result']['dimensions']
        return [
            (
                # Multiple Response and Categorical Array variables have
                # two subsequent dimensions (elements and selections). For
                # this reason it's necessary to pass in both of them in the
                # Dimension class init method. This is needed in order to
                # determine the correct type (CA or MR). We only skip the
                # two-argument constructor for the last dimension in the list
                # (where it's not possible to fetch the subsequent one).
                Dimension(entry)
                if i + 1 >= len(entries)
                else Dimension(entry, entries[i + 1])
            )
            for (i, entry) in enumerate(entries)
        ]
