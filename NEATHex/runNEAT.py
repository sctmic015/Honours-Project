import neat
import neat.nn
from hexapod.controllers.testingNeat import Controller, tripod_gait, reshape
from hexapod.simulator import Simulator
import numpy as np


## x feedforward neuralnet
def evaluate_gait(genomes, config, duration=5):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        leg_params = np.array(tripod_gait).reshape(6, 5)
        #print(net.values)
        try:
            controller = Controller(leg_params, body_height=0.15, velocity=0.46, period=1.0,crab_angle=-np.pi / 6, ann=net)
        except:
            return 0, np.zeros(6)
        simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
        #contact_sequence = np.full((6, 0), False)
        for t in np.arange(0, duration, step=simulator.dt):
            try:
                simulator.step()
            except RuntimeError as collision:
                fitness = 0, np.zeros(6)
        # contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
        fitness = simulator.base_pos()[0]  # distance travelled along x axis
        # summarise descriptor
        #descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        simulator.terminate()
        genome.fitness = fitness


config = neat.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                     neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(False))

winner = p.run(evaluate_gait, 10)

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

print('\nBest genome:\n{!s}'.format(winner))

controller = Controller(tripod_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57, ann = winner_net, printangles= True)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])

while True:
	simulator.step()