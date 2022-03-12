from textwrap import indent
from matplotlib import artist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.animation as animation
import cv2
import glob
import os
from pprint import pprint

from djsp_logger import DJSP_Logger

class Plotter(object):
    def __init__(self, logger):
        self.logger = logger
        # self.num_jobs = len(self.logger.jobs_to_schedule)
        self.makespan = 1000

    def plot_job_to_schedule(self, fig_name):
        cmap = plt.cm.get_cmap('Set1', self.logger.num_job_type + 1)
        yticks = [(i + 1) * 5 for i in range(self.num_jobs)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(0, 5 * self.num_jobs + 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["Job {}".format(i) for i in range(self.num_jobs)])
        ax.set_xlabel('Time')
        max_due_date = 0

        data, color, op_id = [], [], []
        for job_info in self.logger.jobs_to_schedule:
            arrival_time = job_info['arrival_time']
            due_time = job_info['due_time']
            RPT = due_time - arrival_time
            job_type = job_info['job_type']
            data.append([(arrival_time, RPT)])
            color.append(cmap(job_type))
            max_due_date = max(max_due_date, due_time)
        # print(data, color, op_id)
        ax.set_xlim(0, max_due_date+100)
        ax.set_xticks(np.arange(0, max_due_date+100, step=100))
        labels = ['Job Type {}'.format(t) for t in range(self.logger.num_job_type)]
        for i, job_info in enumerate(self.logger.jobs_to_schedule):
            ax.broken_barh(
                data[i], (yticks[i] - 2, 4), 
                facecolors=color[i], edgecolor='black', 
                label=labels[job_info['job_type']])
            labels[job_info['job_type']] = '_nolegend_'
        plt.title('Job Arrival Time & Due Time')
        plt.grid()
        # plt.legend()
        plt.savefig(fig_name)

    def plot_scheduling_process(self, fig_dir):
        # plot scheduling process frames
        machine_histories = [[] for _ in range(self.logger.num_machine)]
        
        for timestep in range(1, len(self.logger.history)):
            for op_info in self.logger.history[:timestep]:
                machine_histories[op_info["machine_id"]].append(op_info)
            cmap = plt.cm.get_cmap('Set1', self.logger.num_job_type + 1)
            yticks = [(i + 1) * 5 for i in range(self.logger.num_machine)]
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_ylim(0, 5 * self.logger.num_machine + 5)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["machine {}".format(i) for i in range(self.logger.num_machine)])
            ax.set_xlim(0, self.makespan+10)
            ax.set_xticks(np.arange(0, self.makespan+10, step=100))
            ax.set_xlabel('Time')
            for i, machine_history in enumerate(machine_histories):
                data, color, op_id = [], [], []
                for op_info in machine_history:
                    data.append((op_info["startTime"], op_info["RPT"]))
                    color.append(cmap(op_info['job_type']))
                    op_id.append(op_info["op_id"])
                    ax.text(
                        x=(op_info["startTime"]+op_info['finishTime'])/2-8, 
                        y=yticks[i], 
                        s='O{},{}'.format(op_info['job_id'], op_info['op_id']), 
                        rotation=270, fontsize=10)
                color = tuple(color)
                ax.broken_barh(data, (yticks[i] - 2, 4), facecolors=color, edgecolor='black')
            plt.savefig('{}/{}.png'.format(fig_dir, timestep))
            plt.close()
    
    def images_to_video(self, fig_dir):
        # print('images_to_video')
        import glob
        img_array = []
        for filename in glob.glob('./scheduling_process/*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter('scheduling_process.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def plot_interactive_gantt(self, htmlname, color_by='job_type'):
        ### timeline
        ### different color means different 'job type'
        ### x-axis: date
        data = self.logger.to('interactive_gantt_input')
        df = pd.DataFrame(data)
        fig = px.timeline(
            df, x_start='StartDateTime', x_end='FinishDateTime', y='machine_id', color=color_by, 
            hover_name='Order', hover_data=['job_id', 'op_id', 'RPT', 'Start', 'Finish'])
        fig.update_layout(xaxis_type='date')    # ['-', 'linear', 'log', 'date', 'category', 'multicategory']
        fig.write_html(htmlname)
    def naive_gantt(self, fig_path):
        cmap = plt.cm.get_cmap('hsv', self.logger.num_job + 1)
        yticks = [(i + 1) * 5 for i in range(self.logger.num_machine)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(0, 5 * self.logger.num_machine + 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["machine {}".format(i) for i in range(self.logger.num_machine)])
        # ax.set_xlim(0, self.makespan())
        ax.set_xlabel('Time')
        machine_histories = self.logger.to('naive_gantt_input')
        for i, history in enumerate(machine_histories):
            data, color = history
            ax.broken_barh(data, (yticks[i] - 2, 4), facecolors=color, edgecolor='black')
            for x, d in enumerate(data):
                ST, RPT = d
                # ax.text(x=ST + RPT/2, 
                #     y=yticks[i],
                #     ha='center', 
                #     va='center',
                #     color='black',
                #    )
        plt.savefig(fig_path)

if __name__ == '__main__':
    logger = DJSP_Logger()
    logger.load('test/test_scheduling.pth')

    pprint(logger.history)
    plotter = Plotter(logger)
    plotter.plot_interactive_gantt('test/interactive_timeline.html')

    # plotter = Plotter(logger)
    # plotter.plot_job_to_schedule('job_to_schedule.png')

    # plotter.plot_scheduling_process('scheduling_process')
    # plotter.images_to_video('scheduling_process')



