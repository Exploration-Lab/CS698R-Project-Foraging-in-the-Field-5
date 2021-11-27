import gym 
import numpy as np
import time
from agent import Agent
import cv2
from PIL import ImageGrab
import neat
import matplotlib.pyplot as plt
import pickle 
import visualize
from neat.math_util import softmax

env = gym.make('my_env:foraging-v0')
imgarray = []
# def screen_record(): 
#     last_time = time.time()
#     printscreen =  np.array(ImageGrab.grab(bbox=(0,40,1920,1080)))
#     # print('loop took {} seconds , shape = {}'.format(time.time()-last_time, printscreen.shape))
#     last_time = time.time()
#     cv2.imwrite('scr.png', printscreen)

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        env.reset()
        ac = np.random.randint(0, 8)
        prev_x = 800
        prev_y = 500
        inx, iny, inc = env.observation_space.shape
        inx = 100
        iny = 100
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fiteness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        dist_max = 0
        done = False
        start_time = time.time()
        while not done:
            img, next_state, tim = env.render()
            # if np.max(final_img) != 0:
            #     final_img = final_img/np.max(final_img)
            # print(z)
            # screen_record()
            # img = cv2.imread('screenshot.png')
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # final_img = cv2.resize(img_gray, (inx, iny),
            #     interpolation = cv2.INTER_NEAREST)
            # print(final_img.shape)
            # cv2.imwrite('scr2.png', final_img)
            # ob = np.reshape(final_img, (inx, iny))
            
            # if density < 2:
            #     imgarray.append(-1)
            next_state = list(next_state)
            next_state.append(juice)
            # next_state.append(tim/1000)
            next_state = np.array(next_state)

            nnOutput = net.activate(next_state)
            # print(nnOutput)
            # softmax_res = softmax(nnOutput)
            # r = np.argmax(((softmax_res/np.max(softmax_res))==1).astype(int))
            r = np.argmax(nnOutput)
            corr, juice, done, info, q = env.step(r)
            xpos = corr[0]
            ypos = corr[1]
            imgarray.clear()
            dist = np.sqrt((xpos - 800)**2 + (ypos - 500)**2)
            # if dist > dist_max:
            fitness_current += 100*info*100*info
            # print(info, q)
                # dist_max = dist
            if xpos == prev_x and ypos == prev_y:
                fitness_current -= 10

            if fitness_current > current_max_fiteness:
                current_max_fiteness = fitness_current
                counter = 0
            else:
                counter += 1

            if done:
                done = True
                fitness_current /= dist*0.001
                # print(genome_id, fitness_current, nnOutput)
                print("ID = {}, Fitness = {}, distance = {}".format(genome_id, fitness_current, dist))
            genome.fitness = fitness_current
            # visualize.plot_stats(pop.statistics)
            # visualize.plot_species(pop.statistics)

# print(inx, iny, inc)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward')

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-20')
winner = p.run(eval_genomes)

with open('winner.okl', 'wb') as output:
    pickle.dump(winner, output, 1)  
    output.close()
