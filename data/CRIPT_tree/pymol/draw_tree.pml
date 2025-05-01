load structure.pdb
load tree.pdb
hide cartoon, tree




#MAKE BONDS FOR TREE EDGES
cmd.bond('tree and id 1', 'tree and id 2')
cmd.bond('tree and id 1', 'tree and id 3')
cmd.bond('tree and id 1', 'tree and id 9')
cmd.bond('tree and id 3', 'tree and id 4')
cmd.bond('tree and id 4', 'tree and id 5')
cmd.bond('tree and id 5', 'tree and id 6')
cmd.bond('tree and id 6', 'tree and id 7')
cmd.bond('tree and id 7', 'tree and id 8')





#COLOR DEFINITIONS
cmd.set_color('I_1_2', (0.7831712152651346, 0.6634790940463787, 0.32969726470869526))
cmd.set_color('I_1_3', (0.8292275089883606, 0.7753832051359706, 0.4417727747802381))
cmd.set_color('I_1_9', (0.8972283335855062, 0.931745683809675, 0.6311075587918794))
cmd.set_color('I_3_4', (0.8151208269598283, 0.7405329639445878, 0.40383288429865666))
cmd.set_color('I_4_5', (0.7776076486673221, 0.6508776865233488, 0.3192159935965749))
cmd.set_color('I_5_6', (0.7794750283869653, 0.6550685970372219, 0.3226422224503434))
cmd.set_color('I_6_7', (0.8310063135234079, 0.7797629900031864, 0.4466914503101059))
cmd.set_color('I_7_8', (0.8567083921128474, 0.8415855961633933, 0.518951218554577))





#BOND COLOR SETTINGS
cmd.set_bond('stick_color', 'I_1_2', 'tree and id 1', 'tree and id 2')
cmd.set_bond('stick_color', 'I_1_3', 'tree and id 1', 'tree and id 3')
cmd.set_bond('stick_color', 'I_1_9', 'tree and id 1', 'tree and id 9')
cmd.set_bond('stick_color', 'I_3_4', 'tree and id 3', 'tree and id 4')
cmd.set_bond('stick_color', 'I_4_5', 'tree and id 4', 'tree and id 5')
cmd.set_bond('stick_color', 'I_5_6', 'tree and id 5', 'tree and id 6')
cmd.set_bond('stick_color', 'I_6_7', 'tree and id 6', 'tree and id 7')
cmd.set_bond('stick_color', 'I_7_8', 'tree and id 7', 'tree and id 8')





#TREE EDGE RADIUS SETTINGS
cmd.set_bond('stick_radius', 0.4747240887935291, 'tree and id 1', 'tree and id 2')
cmd.set_bond('stick_radius', 0.1, 'tree and id 1', 'tree and id 3')
cmd.set_bond('stick_radius', 0.1, 'tree and id 1', 'tree and id 9')
cmd.set_bond('stick_radius', 0.1, 'tree and id 3', 'tree and id 4')
cmd.set_bond('stick_radius', 0.4783164225326146, 'tree and id 4', 'tree and id 5')
cmd.set_bond('stick_radius', 0.47642319998502686, 'tree and id 5', 'tree and id 6')
cmd.set_bond('stick_radius', 0.1, 'tree and id 6', 'tree and id 7')
cmd.set_bond('stick_radius', 0.1, 'tree and id 7', 'tree and id 8')





#q-values set to entropy for spectrum
cmd.alter('resi 0 and name CA', 'q=2.972865901976011')
cmd.alter('resi 0 and name CA and tree', 'vdw=1.4378881566201054')
cmd.alter('resi 1 and name CA', 'q=2.7673736638521618')
cmd.alter('resi 1 and name CA and tree', 'vdw=1.403963879209037')
cmd.alter('resi 2 and name CA', 'q=2.9631080262897678')
cmd.alter('resi 2 and name CA and tree', 'vdw=1.4363132326818362')
cmd.alter('resi 3 and name CA', 'q=2.0556653894114723')
cmd.alter('resi 3 and name CA and tree', 'vdw=1.2715032501137578')
cmd.alter('resi 4 and name CA', 'q=2.953628663586646')
cmd.alter('resi 4 and name CA and tree', 'vdw=1.4347799463606754')
cmd.alter('resi 5 and name CA', 'q=3.0526081854065055')
cmd.alter('resi 5 and name CA and tree', 'vdw=1.450631225036709')
cmd.alter('resi 6 and name CA', 'q=2.458891171299263')
cmd.alter('resi 6 and name CA and tree', 'vdw=1.3497285547151288')
cmd.alter('resi 7 and name CA', 'q=2.329548361304308')
cmd.alter('resi 7 and name CA and tree', 'vdw=1.3256348421077122')
cmd.alter('resi 8 and name CA', 'q=1.3765098787734402')
cmd.alter('resi 8 and name CA and tree', 'vdw=1.1123969194627896')
set cartoon_cylindrical_helices, 1
spectrum q, algae, name CA, 0, 3.5
show sticks, tree
show spheres, tree
set cartoon_transparency, 0.6
set sphere_transparency, 0.3, all
set cartoon_putty_radius, 0.25
set cartoon_putty_transform, 4
set cartoon_putty_scale_power, 1.
set cartoon_putty_scale_min, -1
set cartoon_putty_scale_max, -1
orient
zoom visible, complete=1
