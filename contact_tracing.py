import numpy as np
from random import random
from random import seed as seed_rn
from numpy.random import normal,seed
from numpy.random import choice
from numpy.random import rand
import pylab
from scipy.stats import binned_statistic
import csv
from collections import deque,defaultdict
from time import time

# ------------- TIME-RELATED PARAMETERS -------------------------

day=24*60*60 # one day in seconds

timestep_in_data=300.0 # time steps in data are 300 sec each

# ---------- DEFAULT INTERVENTION PARAMETERS --------------------

p_app_d=0.00      # probability of using a tracking smartphone app
p_tested_d=0.5    # probability of getting tested if with mild symptoms (asymptomatic never get tested; those w severe symptoms always get)
p_traced_d=0.75   # probability of a contact being recalled correctly when contact tracing, set to 0 if no contact tracing
p_mask_d=0.0      # probability of wearing a mask (NOT USED IN THE PAPER)

test_delay_d=0.5*day # delay from onset of symptoms to getting test results
trace_delay_manual_d=1.0*day # additional delay of contact tracing before quarantining contacts
trace_delay_app_d=0.0 # -"- but for apps

manual_tracing_threshold=2 # total number of timeslots for contact to be considered for tracing
app_tracing_threshold=2 # the same

oddweeks_d=False   # set to True for the Interleaving intervention (50% of students on campus on odd weeks and home on even, and vice versa)

mask_reduction_out_d=0.6 # how much transmission probability is reduced, if the infectious student wears a mask
mask_reduction_in_d=0.9   # how much -"-, if the susceptible student wears a mask

tracelength_d=day*2 # store contacts for this long, then remove

quarantine_length_d=day*14 # duration of quarantine

default_intervention_params={"p_app":p_app_d,"p_tested":p_tested_d,"p_traced":p_traced_d,"p_mask":p_mask_d,"test_delay":test_delay_d,"trace_delay_manual":trace_delay_manual_d,"trace_delay_app":trace_delay_app_d,"manual_tracing_threshold":manual_tracing_threshold,"app_tracing_threshold":app_tracing_threshold,"oddweeks":oddweeks_d,"mask_reduction_out":mask_reduction_out_d,"mask_reduction_in":mask_reduction_in_d,"tracelength":tracelength_d,"quarantine_length":quarantine_length_d}

# ------------- SEIR PARAMETERS --------------------------------
# --
# ---- from Report #10 at https://www.epicx-lab.com/covid-19.html

incubation_period=5.2*day # time from transmission to symptoms

prodromal_period=1.5*day # infectiousness begins this many days before symptoms

latency_period=incubation_period-prodromal_period

p_asymptomatic=0.2 # p of an infected individual being asymptomatic

p_paucisymptomatic=0.2*(1-p_asymptomatic)

p_mildsymptoms=0.7*(1-p_asymptomatic)

p_severesymptoms=0.1*(1-p_asymptomatic)

I_classes=['Ias','Ips','Ims','Iss']
I_probs=[p_asymptomatic,p_paucisymptomatic,p_mildsymptoms,p_severesymptoms]

infectious_period=7.5*day-incubation_period

infectiousness_damping=0.51 # for Ias,Ips and their pre-symptomatic period (Ip)

# ---- files etc

#inputfile='including_student_0_positive_rssi_removed.csv'
inputfile='bt_symmetric.csv' # USE ORIGINAL FILE FROM THE SCIENTIFIC DATA PAPER https://www.nature.com/articles/s41597-019-0325-x

path='/m/cs/scratch/networks/jsaramak/spreading' # REPLACE WITH YOUR OWN PATH TO DATA

# ---------- NODE CLASS DEFINITION ----------------

# usage: upon initialization, set to state S
# some properties are randomly chosen (has app, wears mask, etc)
#
# when exposed, call Node.exposure(event_queue,current_time)
# this computes a random timeline to Ip, I, R
# and adds these times to the event queue dictionary
# it also computes whether the student will be tested
# and if, places the beginning of the quarantine in the event queue too

class Node:

    def __init__(self,params,myid=0,currtime=0):

        self.state='S'              # S,E,I,R
        self.infectious=False       # True if the student can infect others
        self.dampingfactor=1.0      # set to <1.0 for asymptomatics etc
        self.contacts={}            # tracedcontacts (dict[student_id]=[time_1,time_2])
        self.contact_has_app=set()  # set of contacts who are known to use app, only used if self has app
        self.in_quarantine=False    # True: doesn't affect anyone else's state
        self.id=myid                # student id
        self.oddweek=int(round(random())) # 0 or 1, meaning present on odd or even weeks, for the interleaving strategy

        if random()<params['p_mask']: # whether the student wears a mask

            self.has_mask=True
            self.mask_factor_out=params['mask_reduction_out']
            self.mask_factor_in=params['mask_reduction_in']

        else:

            self.has_mask=False
            self.mask_factor_out=1.0
            self.mask_factor_in=1.0

        if random()<params['p_app']:

            self.has_app=True

        else:

            self.has_app=False

    def reset(self,myid=0,currtime=0):

        self.state='S'
        self.infectious=False
        self.dampingfactor=1.0
        self.contacts=[]
        self.in_quarantine=False
        self.id=myid
        
    def statechange(self,eventq,newstate,currtime,params,students_with_apps): # changes the state of the student and modifies event dict accordingly

        if newstate=='EOQ':

            self.in_quarantine=False

        elif newstate=='BOQ' or newstate=='BOQ_t':

            if not(self.in_quarantine):

                self.set_quarantine(eventq,currtime,params) # sets self in quarantine and adds end of quarantine to event dict; also does contact tracing

        elif newstate=='CT':

            self.trace_contacts(eventq,currtime,params,students_with_apps) # traces contacts and places (some of) them in quarantine

        else:

            self.state=newstate

            if self.state in ['Ip','Ias','Ips','Ims','Iss']: # infectious only in these states

                self.infectious=True

            else:

                self.infectious=False
       
    def exposure(self,eventq,currtime,params): # sets the student to the exposed state and modifies the timeline, precomputing times to other states

        self.state='E'

        # calculate time to I_p

        time_to_ip=int(timestep_in_data*round((currtime+normal(loc=latency_period,scale=latency_period/10.0))/timestep_in_data))

        eventq[time_to_ip].append((self.id,'Ip'))
        
        # calculate time to I

        time_to_i=int(timestep_in_data*round((time_to_ip+normal(loc=prodromal_period,scale=prodromal_period/10.0))/timestep_in_data))

        # choose Iclass (asymptomatic, pausisymptomatic, mild symptoms, ...)

        Iclass=choice(I_classes,1,p=I_probs)[0]

        eventq[time_to_i].append((self.id,Iclass))

        # set infectiousness factor

        if Iclass=='Iss':

            self.dampingfactor=1.0

        else:

            self.dampingfactor=0.5

        # calculate if tested and quarantined, add quanrantine if so
        # (asymptotics are never tested)


        if not(Iclass=='Ias'):

            if Iclass=='Iss' or random()<params['p_tested']:

                time_to_testing=int(timestep_in_data*round((time_to_i+normal(loc=params['test_delay'],scale=params['test_delay']/10.0))/timestep_in_data))
                       
                if not(self.in_quarantine):

                    eventq[time_to_testing].append((self.id,'BOQ')) # tested and goes to quarantine at time_to_testing

                eventq[time_to_testing+int(timestep_in_data)].append((self.id,'CT')) # do contact tracing when quarantine begins


        # calculate time to H/ICU/R; for our purposes the same since all are removed from the contact network

        time_to_r=int(timestep_in_data*round((time_to_i+normal(loc=infectious_period,scale=infectious_period/10.0))/timestep_in_data))

        eventq[time_to_r].append((self.id,'R'))


    def add_contact(self,student_id,curr_time,tracelength): # adds one contact (time,id) to the contact trace

        self.contacts[student_id].append(curr_time)

    def flush_contacts(self,curr_time,params): # removes all contacts that are older than tracelength

        for contact in self.contacts: # loop thru each contact's list of contact times, pop every contact time that's too old

            while len(self.contacts[contact])>0 and (curr_time-self.contacts[contact][0])>params['tracelength']: 

                self.contacts[contact].popleft()

    def trace_contacts(self,eventq,curr_time,params,students_with_apps): # contact tracing.

        for contact in self.contacts:

            while len(self.contacts[contact])>0 and self.contacts[contact][0]<(curr_time-params['tracelength']): 

                self.contacts[contact].popleft()  

            # check if manual works

            put_in_quarantine=False

            if len(self.contacts[contact])>params['manual_tracing_threshold']: # if present in at least this many 5-min time slots

                if random()<params['p_traced']: # contact detected/recalled correctly w probability p_traced

                    put_in_quarantine=True
                    quarantine_time=int(timestep_in_data*round((curr_time+normal(loc=params['trace_delay_manual'],scale=params['trace_delay_manual']/10.0))/timestep_in_data))

            # check if app-based works if app and manual didn't

            if self.has_app and (contact in students_with_apps) and len(self.contacts[contact])>params['app_tracing_threshold'] and not(put_in_quarantine):

                put_in_quarantine=True
                quarantine_time=int(timestep_in_data*round((curr_time+normal(loc=params['trace_delay_app'],scale=params['trace_delay_app']/10.0))/timestep_in_data))

            # if either worked, quarantine

            if put_in_quarantine:

                if quarantine_time in eventq:

                    if not((contact,'BOQ_t') in eventq[quarantine_time]): # to avoid adding multiple beginnings (if already added from some other contact list)

                        eventq[quarantine_time].append((contact,'BOQ_t')) # place contact in quarantine

                else:

                        eventq[quarantine_time]=[(contact,'BOQ_t')]
       
    def set_quarantine(self,eventq,curr_time,params,trace=False): 

        '''Sets self.in_quarantine=True and adds an EOQ event in event_q after quarantine_length'''

        eoq_time=int(timestep_in_data*round((curr_time+params['quarantine_length'])/timestep_in_data)) # time when quarantine ends

        eventq[eoq_time].append((self.id,'EOQ'))

        self.in_quarantine=True

 
# --------------- AUX FUNCTIONS ----------------

def read_contacts(filename=inputfile): # reads the contact event list to dict[timestamp]=[(node_i,node_j),...]

    '''Returns a dict of events eventdict[timestamp]=[(node_1,node_2),..]
       and a set of student id's encountered in all events.'''

    fn=open(path+filename,'rU')

    f=csv.reader(fn,delimiter=',')

    contactdict={}
    student_ids=set()

    f.next() # skip header line

    while True:

        try:

            line=f.next()

            timestamp=int(line[0])
            node_i=int(line[1])
            node_j=int(line[2])
            signalstrength=int(line[3])

            if node_j>=0:

                # add any filter rules here if req'd

                if timestamp in contactdict:

                    contactdict[timestamp].append((node_i,node_j))

                else:

                    contactdict[timestamp]=[(node_i,node_j)]

                student_ids.add(node_i)
                student_ids.add(node_j)
    
        except:

            break

    return contactdict,student_ids

def read_cluster(filename_root,filename_upto,datapath='your_path_here',normalizer=692.0):

    '''Reads and averages data written by episizes_tracing_cluster into dictionaries''' 

    epidict=defaultdict(float) # epidemic sizes
    redudict=defaultdict(float) # epidemic reduction
    qdict=defaultdict(float) # fraction in quarantine
    fpdict=defaultdict(float) # fraction non-infected in quarantine
    Ndict=defaultdict(float) # N (for normalizing averages)

    for i in xrange(0,filename_upto+1):

        filename=filename_root+"_"+str(i)+".out"
 
        fn=open(datapath+filename,'rU')

        f=csv.reader(fn,delimiter='\t')

        while True:

            try:

                line=f.next()

                if not(line[0]=='Parameter' or line[0][1]=='T'):

                    p_trace=round(float(line[0]),2)
                    p_app=round(float(line[1]),2)

                    Ndict[(p_trace,p_app)]+=1.0 # all dictionaries have value pairs as keys (prob of manual tracing, prob of app use)
                    epidict[(p_trace,p_app)]+=float(line[2])
                    qdict[(p_trace,p_app)]+=float(line[3])
                    fpdict[(p_trace,p_app)]+=float(line[4])

            except:

                break

    maxsize=epidict[(0.0,0.0)]

    for key in Ndict:

        redudict[key]=1.0-epidict[key]/maxsize

        epidict[key]/=(normalizer*Ndict[key])
        qdict[key]/=(normalizer*Ndict[key])
        fpdict[key]/=Ndict[key]

    return epidict,redudict,qdict,fpdict

# ---------------- RUNNING MULTIPLE RUNS

def episizes_tracing_cluster(contactdict,student_ids,params,iterations=10):

    '''Runs iterations run of the SEIR model with the CH data using parameters defined in params,
    over the ranges of app probabilities and manual contact tracing probabilities defined below.
    Developed for parallel runs using a cluster, so prints out all results so that they can be
    piped into a file and read later (example reader code is above)'''

    app_probabilities=np.arange(0.0,1.1,0.1)
    trace_hitrates=np.arange(0.0,1.1,0.1)

    datalist=[]
    qlist=[]
    fplist=[]

    t1=time()

    # to speed up things, compute the time of the first event of each student and place in dictionary

    first_times={}

    times=sorted(contactdict.keys())

    for curr_time in times:

        for contact in contactdict[curr_time]:

            node_i=contact[0]
            node_j=contact[1]

            if not(node_i in first_times):

                first_times[node_i]=curr_time

            if not(node_j in first_times):

                first_times[node_j]=curr_time

        if len(first_times)==len(student_ids):

            break

    for parameter in params:

        print "Parameter\t"+parameter+"\t"+str(params[parameter])

    for pt in trace_hitrates:

        for ap in app_probabilities:

            params['p_app']=ap
            params['p_traced']=pt

            Ilist=[]
            qlist=[]
            fplist=[]

            for i in xrange(0,iterations):

                I,q,fp=SEIR_onerun_grid(contactdict,student_ids,first_times=first_times,params=params)

                Ilist.append(I)
                qlist.append(q)
                fplist.append(fp)

            for i in xrange(0,len(Ilist)):

                I=Ilist[i]
                q=qlist[i]
                fp=fplist[i]

                print str(pt)+"\t"+str(ap)+"\t"+str(I)+"\t"+str(q)+"\t"+str(fp)

    print "Time: "+str((time()-t1)/60.0)+" min"

def SEIR_onerun_grid(contactdict,student_ids,params=default_intervention_params,first_times={},p_transmission=0.00625,initial_period_in_days=7,I_only=False,chain=False,dailynets=False,nets_per_day=8,animate=False,curr_layout=[],R0=False):

    '''Does one run of the SEIR model, returning various items depending on user choice (see end of function).
       Required inputs:
                contactdict: dictionary[time]=[contact1,contact2,contact2] where contact=(student_id1,student_id2)
                student_ids: set of all student ids in contactdict
       User choices:
                first_times: dictionary[student_id]=timestamp of student_id's first contact (speeds up runs if precomputed)
                p_transmission: probability of transmission from I to S in one timestep, default p=0.004
                initial_period_in_days: determines infection time of first patient (first contact time + randomly drawn time from initial_period_in_days)
                I_only: if True, the run only returns the final number of infected 
                chain: if True, the run also returns the transmission chain as an event list [(source1,target1,time1),(source2,...]
                dailynets: if True, the run also returns for each period a pynet.SymmNet() network + dictionaries of student states + transmission list
                nets_per_day: (default=8) how many nets to return per day in dailynets and how many frames per day for animations
                animate: if True, the run only returns a list of frames to be animated with moviepy.editor.ImageSequenceClip()
                curr_layout=[]: only used for animation, if one wants to start from a pre-determined node layout
        Outputs:
                depend on the above choices. See end of function'''

    student_id_list=list(student_ids) # list of ids in data; not all ids appear at all

    N_students=len(student_id_list)

    max_time=max(contactdict.keys())

    patient_zero_index=choice(student_id_list)

    # initialize all students
    # so that studentlist[i] = class Node in S state

    studentlist={}
    students_with_apps=set()

    for sid in student_id_list:

        studentlist[sid]=Node(params=params,myid=sid,currtime=0)
        if studentlist[sid].has_app:

            students_with_apps.add(sid)

        for sid2 in student_id_list:

            studentlist[sid].contacts[sid2]=deque() # queue with fast left pop

    if len(first_times)==0:

        first_times={}

        times=sorted(contactdict.keys())

        for curr_time in times:

            for contact in contactdict[curr_time]:

                node_i=contact[0]
                node_j=contact[1]
    
                if not(node_i in first_times):
    
                    first_times[node_i]=curr_time

                if not(node_j in first_times):

                    first_times[node_j]=curr_time

            if len(first_times)==len(studentlist):

                break

    # initialize event queue

    eventq=defaultdict(list) # dictionary dict[timestamp]={(student_id,state),...] of state changes

    # find first event where patient zero participates; start from there.

    curr_time=first_times[patient_zero_index] # pick first event of patient zero as starting time

    curr_time=curr_time+int(timestep_in_data*round((day*random()*initial_period_in_days)/timestep_in_data)) # add 0-7 days at random

    studentlist[patient_zero_index].exposure(eventq,curr_time,params) # now expose patient zero

    exposed=1
    infectious=0
    total_infected=1
        
    done=False
    quarantines=0
    false_quarantines=0

    periodic_boundary_modifier=0

    while not (done):

        # --------- handle event queue

        if curr_time in eventq:

            for event in eventq[curr_time]: # eventq events: (student_id,new_state)

                if event[1]=='BOQ_t' and not(studentlist[event[0]].in_quarantine):

                    quarantines+=1

                    if studentlist[event[0]].state=='S' or studentlist[event[0]].state=='R':

                        false_quarantines+=1

                elif event[1]=='R': # book-keeping for the various classes (S=susceptible,E=exposed,Ip=presymptomatic infectious,I=infectious, Ic=cumulative number of infected, R=recovered)

                    infectious-=1

                elif event[1][0]=='I':

                    exposed-=1
                    infectious+=1

                studentlist[event[0]].statechange(eventq,event[1],curr_time,params,students_with_apps) # change student's state (disease progression, quarantine)

            del eventq[curr_time]

        # --------- handle transmission and contacts

        if (curr_time-periodic_boundary_modifier)>max_time:

            periodic_boundary_modifier+=max_time+300

        if (curr_time-periodic_boundary_modifier) in contactdict: # if there are contacts at time t=curr_time

            for contact in contactdict[(curr_time-periodic_boundary_modifier)]: # loop through all contacts that take place at t=curr_time

                if not(studentlist[contact[0]].state=='S' and studentlist[contact[1]].state=='S') and not(studentlist[contact[0]].in_quarantine or studentlist[contact[1]].in_quarantine): # note: works only for this particular SEIR timeline (incubation time etc)

                    studentlist[contact[0]].contacts[contact[1]].append(curr_time)
                    studentlist[contact[1]].contacts[contact[0]].append(curr_time)

                    if (studentlist[contact[0]].infectious and studentlist[contact[1]].state=='S') or (studentlist[contact[1]].infectious and studentlist[contact[0]].state=='S'):

                        if studentlist[contact[1]].state=='S':

                            source=contact[0]
                            target=contact[1]

                        else:

                            source=contact[1]
                            target=contact[0]

                        if random()<p_transmission*studentlist[source].dampingfactor:# *studentlist[source].mask_factor_out*studentlist[target].mask_factor_in: # source infects target with this probability

                            studentlist[target].exposure(eventq,curr_time,params) # expose the target and recompute event queue to add disease progression events for target

                            exposed+=1
                            total_infected+=1
                
            if (exposed+infectious)==0: # simulation done when no-one is infectious or exposed

                done=True

            curr_time+=int(timestep_in_data)


        # -------------- done looping over contacts at time curr_time

#    print time()-t1

    if quarantines>0:

        fq=float(false_quarantines)/quarantines

    else:

        fq=0.0

    return total_infected,quarantines,fq # only return the final total number of infected + ppl in quarantine + false positive ratios

if __name__=="__main__":

    contactdict,student_ids=read_contacts()

    params=default_intervention_params

    params['manual_tracing_threshold']=2
    params['app_tracing_threshold']=2
    params['p_tested']=0.5

    episizes_tracing_cluster(contactdict,student_ids,params,iterations=100)

    

    






                

            

    



        

    

    

    

    

    

                

                

            

            
        

            

        

        

        
    
            


    

        

        
        

        

        

        
