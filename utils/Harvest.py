#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pytesseract
import cv2
from PIL.ImageOps import invert
import time
import win32api, win32con 
from PIL import ImageGrab
import matplotlib.pyplot as plt
import numpy as np
import re


# In[2]:

def sample_around_cell(frame,cell,size):
    return frame[cell[0]-size:cell[0]+size,cell[1]-size:cell[1]+size]
    
def sample_delim(delim):
    im = screen()
    return im[delim[0]:delim[1],delim[2]:delim[3]]

def change_map(x,y,delay):
    change = 0
    im = sample_delim([100,110,0,10])
    counter = 0
    while change < 1:
        im2 = sample_delim([100,110,0,10])
        counter += 1
        if not (np.prod(im==im2) or np.mean(im2)>240):
            change += 1
            im = im2
        if counter > 49:
            break
    time.sleep(delay)
    
def go_to(coordx,coordy):
    return 0

def left_map(delay):
    bias = 75*(np.random.random()-0.5)
    click(0,350+bias)
    change_map(0,350,delay)
    
def right_map(delay):
    bias = 75*(np.random.random()-0.5)
    click(1365,350+bias)
    change_map(1365,350,delay)
    
def up_map(delay):
    bias = 75*(np.random.random()-0.5)
    click(750+bias,0)
    change_map(750,0,delay)
    
def down_map(delay):
    bias = 75*(np.random.random()-0.5)
    click(750+bias,675)
    change_map(750,675,delay)

def plot(im):
    plt.figure(figsize=(15,15))
    plt.imshow(im)
    plt.show()

def click(x,y): 
    win32api.SetCursorPos((x,y))
    time.sleep(0.1+0.05*np.random.random())
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(0.1+0.05*np.random.random())
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0) 
    
def put_in_bank():
    time.sleep(3)
    win32api.SetCursorPos((930,150))
    sleep(0.5)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,930,150,0,0)
    sleep(0.5)
    win32api.SetCursorPos((250,150))
    sleep(0.5)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,250,150,0,0) 
    sleep(0.5)
    click(350,110)
    
def left(cell,k):
    return cell + k*np.array([-15,-30])

def right(cell,k):
    return cell + k*np.array([15,30])

def up(cell,k):
    return cell + k*np.array([-15,30])

def down(cell,k):
    return cell + k*np.array([15,-30])

fight = np.load('fight.npy')

def is_fighting(fight):
    sample = sample_delim([569,573,1348,1359])
    diff = np.array(sample,dtype='int32') - np.array(fight,dtype='int32')
    return np.mean(np.abs(diff))<10

inventory = np.load('inventory.npy')

def is_inventory(inventory):
    sample = sample_delim([55,60,920,925])
    return np.prod(sample==inventory)

merchant = np.load('merchant.npy')

def is_merchant(merchant):
    sample = sample_delim([430,435,750,755])
    return np.prod(sample==merchant)

full = np.load('full.npy')

def is_full(full):
    sample = sample_delim([455,460,20,25])
    return np.prod(sample==full)

locate_player = np.load('locate_player.npy')

def is_last(locate_player):
    sample = sample_delim([580,585,1300,1305])
    diff = np.array(sample,dtype='int32') - np.array(locate_player,dtype='int32')
    return np.mean(np.abs(diff))<10

up_work = np.load('lvl_up_work.npy')
up_work2 = np.load('up_work2.npy')

def is_lvl_up_work(up_work):
    sample = sample_delim([455,465,165,185])
    return np.mean(np.abs(sample-up_work))<40

def sell():
    time.sleep(5)
    click(550,380)
    time.sleep(5)
    click(600,80)
    time.sleep(5)
    click(920,130)
    for _ in range(11):
        time.sleep(3)
        click(350,320)
    time.sleep(3)
    click(1130,50)

def manage_fight():
    #bool_block = True
    bool_last = True
    bool_start = True
    just_fight = False
    while is_fighting(fight):
        just_fight = True
        
        #if bool_block:
        #    bool_last = is_last(locate_player)
        #    bool_block = False
            
        if bool_start:
            time.sleep(0.5)
            click(230,50)
            time.sleep(1)
            click(1050,720)
            bool_start = False
            time.sleep(4)
            
        if bool_last:
            time.sleep(0.5)
            click(230,50)
            time.sleep(0.5)
            click(635,700)
            time.sleep(2)
            click(1250,580)
            time.sleep(3)
            
        else:
            time.sleep(0.5)
            click(230,50)
            time.sleep(0.5)
            click(635,700)
            time.sleep(2)
            click(1280,580)
            time.sleep(3)
            
        if  not is_fighting(fight):
                break
            
        if bool_last:
            time.sleep(0.5)
            click(230,50)
            time.sleep(0.5)
            click(635,700)
            time.sleep(2)
            click(1250,580)
            time.sleep(2)
        else:
            time.sleep(0.5)
            click(230,50)
            time.sleep(0.5)
            click(635,700)
            time.sleep(2)
            click(1280,580)
            time.sleep(2)
            
        if not is_fighting(fight):
                break
            
        time.sleep(0.5)
        click(230,50)
        time.sleep(1)
        click(1050,720)
        
        time.sleep(4)
            
    if just_fight:
            sleep(6)
            click(650,450)
            sleep(6)
            click(1020,700)
            sleep(5)
            click(1020,90)
            sleep(5)
            click(900,150)
            click(900,150)
            sleep(5)
            click(1130,50)
            sleep(4)
            
def manage_fight2():
    bool_start = True
    just_fight = False
    while is_fighting(fight):
        just_fight = True
            
        if bool_start:
            time.sleep(0.5)
            click(230,50)
            time.sleep(1)
            click(1050,720)
            bool_start = False
            time.sleep(4)
        
        #Première attaque
        time.sleep(0.5)
        click(230,50)
        time.sleep(0.5)
        click(605,700)
        time.sleep(2)
        click(1250,580)
        time.sleep(3)
            
        if  not is_fighting(fight):
                break
        
        #Deuxième attaque
        time.sleep(0.5)
        click(230,50)
        time.sleep(0.5)
        click(605,730)
        time.sleep(2)
        click(1250,580)
        time.sleep(3)
        
        #Passer le tour
        time.sleep(0.5)
        click(230,50)
        time.sleep(1)
        click(1050,720)
        
        time.sleep(4)
            
    if just_fight:
            sleep(6)
            click(650,450)
            sleep(6)
            click(1020,700)
            sleep(5)
            click(1020,90)
            sleep(5)
            click(900,150)
            click(900,150)
            sleep(5)
            click(1130,50)
            sleep(4)

def samples_around_cell(frame,cell,size):
    res =  [sample_around_cell(frame,np.array(cell)+np.array([-3,-14]),size)]
    res +=  [sample_around_cell(frame,np.array(cell)+np.array([-3,14]),size)]
    res =  [sample_around_cell(frame,np.array(cell)+np.array([-30,0]),size)]
    res +=  [sample_around_cell(frame,np.array(cell)+np.array([-22,10]),size)]
    res +=  [sample_around_cell(frame,np.array(cell)+np.array([-22,-10]),size)]
    #res +=  [sample_around_cell(frame,np.array(cell)+np.array([-3,0]),size)]
    res += [sample_around_cell(frame,np.array(cell)+np.array([-3,10]),size)]
    return res

def black(frame,cell,size):
    frame[cell[0]-size:cell[0]+size,cell[1]-size:cell[1]+size] = np.zeros((2*size,2*size))
    
def all_black(frame,index,size):
    for ind in index:
        black(frame,np.array(ind)+np.array([-3,-14]),size)
        black(frame,np.array(ind)+np.array([-3,14]),size)
        black(frame,np.array(ind)+np.array([-30,0]),size)
        black(frame,np.array(ind)+np.array([-22,10]),size)
        black(frame,np.array(ind)+np.array([-22,-10]),size)
        black(frame,np.array(ind)+np.array([-3,0]),size)
    return frame
    
def current_index(index,frame1,frame2,thres):
    changed = []
    for ind in index[1:]:
        sample1 = samples_around_cell(frame1,ind,2)
        sample2 = samples_around_cell(frame2,ind,2)
        chang = True
        for k in range(len(sample1)):
            diff_ = np.array(sample2[k],dtype='int32')-np.array(sample1[k],dtype='int32')
            indic = np.mean(np.abs(diff_))
            chang = chang and indic>thres
        if chang:
            changed += [ind]
    return changed

def current_index2(index,frame1,frame2,thres):
    changed = []
    for ind in index[1:]:
        sample1 = sample_around_cell(frame1,ind,15)
        sample2 = sample_around_cell(frame2,ind,15)
        diff_ = np.array(sample2,dtype='int32')-np.array(sample1,dtype='int32')
        indic = np.mean(np.abs(diff_))
        chang = indic>thres
        if chang:
            changed += [ind]
    return changed
         
def screen():
    im = ImageGrab.grab()
    im = np.array(im)
    im = im[:,:,0]
    return im

def sleep(t):
    time.sleep(0.9*t + 0.1*t*np.random.random())
           
def harvest_(empty,index,mov,delay,thres,coord_):
    first_harv = True
    changed = current_index(index,empty,screen(),thres)
    ind_ = [-1,-1]
    while not changed == []:
        coord = get_coord(ImageGrab.grab(),70)
        if not (coord[0] == coord_[0] and coord[0] == coord_[0]):
            return 'not_arrived'
        sleep(0.6)
        ind = changed[0]
        if ind[0] == ind_[0] and ind[1] == ind_[1]:
            if len(changed) == 1:
                break
            else:
                ind = changed[1]
        if is_inventory(inventory):
            sleep(1)
            click(1135,70)
        if is_merchant(merchant):
            sleep(1)
            click(750,430)
        if is_full(full):
            sleep(1)
            click(275,430)
            if not is_lvl_up_work(up_work):
                return 'go_to_bank'
        click(ind[1],ind[0]-30)
        #click(ind[1],ind[0]-25)
        win32api.SetCursorPos((60,60))
        if first_harv:
            sleep(2)
            first_harv = False
        sleep(2.4+np.random.random())
        if len(changed)==1:
            sleep(2)
            #ind = index[0]
            #click(ind[1],ind[0])
            #win32api.SetCursorPos((50,50))
        
        manage_fight()
        changed = current_index(index,empty,screen(),thres)
        ind_ = [ind[0],ind[1]]
    manage_fight()
    return 'finish'

def diff(list1,list2):
    res = []
    for e in list1:
        if not e in list2:
            res += [e]
    return res

def harvest_2(empty,index,mov,delay,thres,coord_):
    first_harv = True
    changed = current_index2(index,empty,screen(),thres)
    ind_ = [-1,-1]
    visited = []
    while not changed == []:
        coord = get_coord(ImageGrab.grab(),70)
        if not (coord[0] == coord_[0] and coord[0] == coord_[0]):
            return 'not_arrived'
        sleep(0.6)
        ind = changed[0]
        visited += [ind]
        if ind[0] == ind_[0] and ind[1] == ind_[1]:
            if len(changed) == 1:
                break
            else:
                ind = changed[1]
        if is_inventory(inventory):
            sleep(1)
            click(1135,70)
        if is_merchant(merchant):
            sleep(1)
            click(750,430)
        if is_full(full):
            sleep(1)
            click(275,430)
            if not is_lvl_up_work(up_work2):
                return 'go_to_bank'
        click(ind[1],ind[0])
        #click(ind[1],ind[0]-25)
        win32api.SetCursorPos((60,60))
        if first_harv:
            sleep(2)
            first_harv = False
        sleep(4+np.random.random())
        if len(changed)==1:
            sleep(2.5)
            #ind = index[0]
            #click(ind[1],ind[0])
            #win32api.SetCursorPos((50,50))
        
        manage_fight2()
        changed = current_index2(index,empty,screen(),thres)
        changed = diff(list(map(list,changed)),list(map(list,visited)))
        ind_ = [ind[0],ind[1]]
    manage_fight2()
    return 'finish'
        

def turn(empty_maps,index_maps,to_hdv,move_between_maps,delay1,delay2):

    harvest2(empty_maps[0],index_maps[0],to_hdv[0],delay2)
    time.sleep(2)
    make_move_maps(move_between_maps[0],delay1)

    harvest2(empty_maps[1],index_maps[1],to_hdv[1],delay2)
    time.sleep(2)
    make_move_maps(move_between_maps[1],delay1)

    harvest2(empty_maps[2],index_maps[2],to_hdv[2],delay2)
    time.sleep(2)
    make_move_maps(move_between_maps[2],delay1)

    harvest2(empty_maps[1],index_maps[1],to_hdv[1],delay2)
    time.sleep(2)
    make_move_maps(move_between_maps[3],delay1)

def get_coord(im,thres):
    area = (10, 35, 72, 55)
    img = invert(im.crop(area))
    img = np.array(img)[:,:,2]
    retval, img = cv2.threshold(img,thres,255, cv2.THRESH_BINARY)
    img = cv2.resize(img,(0,0),fx=3,fy=3)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.medianBlur(img,3)
    img_ = np.zeros((150,300))+255
    img_[45:105,57:243] = img
    txt = pytesseract.image_to_string(img_)
    numbers = re.findall(r'-?\d+', txt)
    coord1 = int(numbers[0])
    coord2 = int(numbers[1])
    return coord1,coord2

def get_price():
    im = ImageGrab.grab()
    area = (350, 465, 395, 480)
    img = invert(im.crop(area))
    img = np.array(img)[:,:,2]
    retval, img = cv2.threshold(img,150,255, cv2.THRESH_BINARY)
    img = cv2.resize(img,(0,0),fx=5,fy=5)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.medianBlur(img,3)
    txt = pytesseract.image_to_string(img)
    numbers = re.findall('[0-9]+', txt)
    price = 1000*int(numbers[0]) + int(numbers[1])
    return price