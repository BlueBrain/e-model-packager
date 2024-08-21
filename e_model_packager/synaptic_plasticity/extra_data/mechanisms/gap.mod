COMMENT
/**
 * @remark Copyright (c) BBP/EPFL 2005-2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
 */
ENDCOMMENT

NEURON {
POINT_PROCESS Gap
RANGE vgap
RANGE g, i
NONSPECIFIC_CURRENT i
RANGE synapseID, selected_for_report
}
PARAMETER {
    g = 1 (nanosiemens)
    selected_for_report = 0
}
ASSIGNED {
v (millivolt)
vgap (millivolt)
i (nanoamp)
synapseID
}
BREAKPOINT { i = (v - vgap)*(g*1e-3) }
