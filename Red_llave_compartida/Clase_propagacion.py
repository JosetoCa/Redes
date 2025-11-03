# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:59:00 2022

@author: elgaral
"""

import numpy as np

class propa:
    
    # class attribute
    version = '0.0.2'
    name = 'Paraxial-FT'
    units = 'Millimeter'
    
    # Instance attribute
    def __init__(self,L = 10.0, N = 1024, wl = 0.532e-3, z = 100.0):
        """
        Parameters
        ----------
        L : float
            The default is 10.0. Side lenght of the square area in mm
        N : TYPE, integer
            DESCRIPTION. The default is 1024. Side lenght of the square matrix
        wl : float
            The default is 0.532e-3. Wavelenght in mm
        z : float
            The default is 100.0. Propagation distance in mm

        Returns
        -------
        None.

        """
        self.L = L
        self.N = N
        self.wl = wl
        self.z = z
        self.H = None
        self.h = None
        self.aprox = None
        
    # Methods
        
    def parameters(self):
        print('--------------------------------------------------------------')
        print('L = {} mm'.format(self.L))
        print('N = {}'.format(self.N))
        print('wl = {} mm'.format(self.wl))
        print('z = {} mm'.format(self.z))
        print('--------------------------------------------------------------')
        
    def resolution(self):
        """
        Basic information about resolution, suggested approximation, and validity
        of the approximation

        Returns
        -------
        aprox : string
            suggested approximation. 'TF' Transfer function, 'IR' Impulse 
            response.
        Ds : float
            Side length of the support area.
        BW : float
            Bandwidth. Maximum reproducible spatial frequency.

        """
        dx = self.L/self.N
        print('--------------------------------------------------------------')
        print('Spatial resolution = L/N \t Regime = wl*z*N/L^2\n')
        print('BW: IR = L/(2*wl*z) \t TF= N/(2*L)')
        print('Spatial resolution = {} mm'.format(dx))
        regime = self.wl*self.z/(dx*self.L)
        print('Regime value = {}'.format(regime))
        if regime < 1:
            print('TF approximation suggested')
            self.aprox = 'TF'
            Ds = self.L/3 + self.wl*self.z/dx
            BW = 1/(2*dx)
        elif regime == 1:
            print('Ideal case, both approximations work')
            self.aprox = 'TF'
            Ds = self.L
            BW = 1/(2*dx)
        else:
            print('IR approximation suggested')
            self.aprox = 'IR'
            Ds = self.L/3
            BW = self.L/(2*self.wl*self.z)
        print('Side length of effective area = {} mm'.format(Ds))
        print('Bandwidth = {} 1/mm'.format(BW))
        print('--------------------------------------------------------------')
        return  Ds , BW
    
    def TFunc(self,salvar='N'):
        """
        Creates the transfer function for the parameters of the object.

        Parameters
        ----------
        salvar : string, optional
            Saves the array in a binary formta npy. The default is 'N' do not save,
            'Y' saves.

        Returns
        -------
        None. Creates de object parameter H

        """
        dx = self.L/self.N
        k = 2*np.pi/self.wl
        fx = np.fft.fftfreq(self.N, d=dx) 
        FX , FY = np.meshgrid(fx,fx)
        self.H = np.exp(1j*k*self.z)*np.exp(-1j*np.pi*self.wl*self.z*(FX**2+FY**2))
        if salvar == 'Y':
            np.save('H_N{}_L{}_wl{}_z{}'.format(self.N,self.L,self.wl,self.z),self.H)
            
    
    def IRfunc(self,salvar='N'):
        """
        Creates the impulse response function for the parameters of the object.

        Parameters
        ----------
        salvar : string, optional
            Saves the array in a binary formta npy. The default is 'N' do not save,
            'Y' saves.

        Returns
        -------
        None. Creates de object parameter h

        """
        dx = self.L/self.N
        k = 2*np.pi/self.wl
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        self.h = (1/(1j*self.wl*self.z))*np.exp(1j*k*self.z) \
        *np.exp(1j*k/(2*self.z)*(X**2+Y**2))
        if salvar == 'Y':
            np.save('h_N{}_L{}_wl{}_z{}'.format(self.N,self.L,self.wl,self.z),self.h)
            
    def propaTF(self,u1):
        """
        Propagates the object using a paraxial aproximation with the transfer
        function method

        Parameters
        ----------
        u1 : float
            Is the object ot be propagated. It must be square.

        Returns
        -------
        u2 : float
            Object propagation result.

        """
        print('--------------------------------------------------------------')
        print('propaTF: matríz {} X {}'.format(self.N,self.N))
        
        U1 = np.fft.fft2(u1)
        u2 = np.fft.ifft2(U1*self.H)
        del U1
        print('Terminó.')
        print('--------------------------------------------------------------')
        return u2
    
    def propaIR(self,u1):
        """
        Propagates the object using a paraxial aproximation with the impulse
        response function method

        Parameters
        ----------
        u1 : float
            Is the object ot be propagated. It must be square.

        Returns
        -------
        u2 : float
            Object propagation result.

        """
        print('--------------------------------------------------------------')
        print('propaIR: matríz {} X {}'.format(self.N,self.N))
        dx = self.L/self.N
        H = np.fft.fft2(np.fft.fftshift(self.h))*dx**2
        U1 = np.fft.fft2(np.fft.fftshift(u1))
        u2 = np.fft.fftshift(np.fft.ifft2(U1*H))
        del H,U1
        print('Terminó.')
        print('--------------------------------------------------------------')
        return u2
    def propa(self, u1):
        """
        Propagates the object using Fresnel aproximation. 
        The method of propagation used is selected through
        resolution instance, the aprox parameter.

        Parameters
        ----------
        u1 : float
            Is the object ot be propagated. It must be square.

        Returns
        -------
        u2 : float
            Object propagation result.
        """
        if self.aprox is None:
            self.resolution()
        if self.aprox == 'TF':
            if self.H is None:
                self.TFunc()
                u2 = self.propaTF(u1)
                return u2
            else:
                u2 = self.propaTF(u1)
                return u2
        else:
            if self.h is None:
                self.IRfunc()
                u2 = self.propaIR(u1)
                return u2
            else:
                u2 = self.propaIR(u1)
                return u2

            