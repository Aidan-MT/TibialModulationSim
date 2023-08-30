"""
Document contains the parameters for the full simulation (i.e., weights)
"""

parameters = [
    9.557082542556262, #First inhibitory connection from Independant -> Relay in PAG
    19.568714569351947, #Then excitatory connection from Relay in PAG to Transient neuron
    8.913873721369256, #From secondary relay (pathway B) to inverse neuron
    12.83068963701047, #Then PAG relay to type I direct neuron
    22.825599522654723, #Then input from spine to relay 1 (excitatory)
    11.759585222652216, #input from spine to relay 2 (excitatory)
    11.019118252602249, #Transient to Inverse neuron
    2.6463420020209334, #type I direct to type II direct
    15.134916357687013, #type II direct to inverse neuron
    6.652336230639002, #Inverse neuron to type I direct
    2.9002218793601418, #tonically active independent neuron to type I direct
    2.2867886978133236, #Onuf to EUS
    3.576068937872625, #EUS to pudendal afferent
    23.466801684076103, #Pudendal afferent to pudendal process
    19.347003514066273, #Pudendal process to PGN stimulator
    27.109113151928813, #Pudendal process to PGN
    6.660785402776283, #Pudendal process to ascending output gate
    12.014232423699482, #PGN stimulator to PGN
    15.122565616782811, #Descending input to PGN
    11.692271838241286, #Descending input to PGN stimulator
    16.858270160247688, #Descending input to Onuf inhibiting interneuron
    26.746233595194834, #Onuf inhibitor to onuf's nucleus
    26.36771490789497, #Hypogastric afferent to hypogastric stimulator
    5.205756193863606, #Hypogastric stimulator to hypogastric nerve
    28.892527748852498, #Hypogastric nerve (preganglionic) to Inferior Mesenteric Ganglia 
    20.016363467967604, #IMG to hypogastric efferent
    9.68584362969697, #Hypogastric efferent to IUS
    20.55601326863998, #descending input to inhibitory interneuron
    20.112831818853813, #inhibitory interneuron to hypograstric nerve (preganglionic)
    10.26819696250778, #Pelvic afferent to ascending output
    12.235103106207598, #Pelvic afferent to pelvic process 
    24.410839719756527, #Pelvic afferent to hypogastric process
    10.606996865936077, #Pelvic process to Onuf's Nucleus
    9.581589928415253, #Hypogastric process to hypogastric nerve 
    25.399721117597117, #PGN to pelvic efferent
    22.667247958558594, #pgn to inhibitory interneuron
    12.246919683621238, #pgn to its reciprocal neuron
    16.308880498981875, #pgn reciprocal to pgn stimulator (inhibitory)
    18.546917656265535, #inhibitory interneuron to ascending gate (inhibitory)
    4.347482900256455, #Descending input to inverse neuron
    19.398036845636188, #Inverse neuron to pgn reciprocal (inhibitory)
    11.40546924064233, #Inverse neuron to inhibitory interneuron (inhibitory)
]