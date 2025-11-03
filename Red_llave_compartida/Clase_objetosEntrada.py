# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:39:49 2022

@author: elgaral
"""
import numpy as np
from scipy.special import jv
from PIL import Image
from scipy.ndimage import map_coordinates
from typing import Tuple

class objPlanoEntrada:

    # Class attribute
    version = '0.0.2'
    name = 'Plane objects'
    units = 'Millimeters'

    # Instance attributes
    def __init__(self, L = 10.0, N = 1024):
        """
        Parameters
        ----------
        L : float, optional
            Side length of working area. The default is 10.0 mm.
        N : integer, optional
            Side length of array. The default is 1024.

        Returns
        -------
        None.

        """
        self.L = L
        self.N = N

    # Methods

    def centerCircle(self,radi,salvar = 'N'):
        """
        Generates a centered iris.

        Parameters
        ----------
        radi : float
            Radius of the iris in mm.
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy.

        Returns
        -------
        circ : np.array NxN.

        """
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        circ = np.zeros((self.N,self.N))
        circ[R<=radi] = 1.0
        if salvar == 'Y':
            np.save('cIris_N{}_L{}_radi{}'.format(self.N,self.L,radi),circ)
        return circ
    
    def centerEmbedded(self,Nx,Ny,u1):
        """
        Embedes a square window-matrix inside a plane-matrix of zeros

        Parameters
        ----------
        Nx : integer.
            Plane-matrix dimension in X-coord (columns).
        Ny : Integer
            Plane-matrix dimension in Y-coord (rows).
        u1 : complex
            Square window-matrix.

        Returns
        -------
        u2 : Complex
            Window-matrix embedded in Plane-matrix.

        """
        Nw = np.shape(u1)[0]
        u2 = np.zeros_like(u1, shape = (Ny,Nx))
        u2[int((Ny-Nw)/2):int((Ny-Nw)/2)+Nw,
                int((Nx-Nw)/2):int((Nx-Nw)/2)+Nw] = u1
        return u2

    def centerFSPP(self,ell,Nell,salvar = 'N'):
        """
        Generates a centered spiral phase plate.

        Parameters
        ----------
        ell : float
            Fractional topological charge of the SPP
        Nell : integer
            Maximun integer topological charge of the expansion
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy

        Returns
        -------
        spp : complex float
            Centered spiral phase mask

        """
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        theta = np.arctan2(Y,X)
        sumSpp = 0 + 1j*0
        for n in range(0,Nell):
            sumSpp = sumSpp + np.exp(1j*n*theta)/(ell-n) + np.exp(-1j*n*theta)/(ell+n)
        spp = np.exp(1j*np.angle(np.exp(1j*ell*np.pi)*np.sin(np.pi*ell)*sumSpp/np.pi))
        if salvar == 'Y':
            np.save('spp_N{}_L{}_ell{}'.format(self.N,self.L,ell),spp)
        return spp

    def centerGauss(self, z=0.0, w0=1.0):
        """
        Complex gaussean beam, with amplitude 1 in the waist.

        Parameters
        ----------
        z : float
            distance from the waist. (mm)
        w0: 
            radius of the waist. (mm)
        
        Returns
        -------
        E : np.array NxN type complex 128.
        """
        x = np.linspace(-self.L/2, self.L/2, self.N)
        X, Y = np.meshgrid(x,x)
        r2 = X**2 + Y**2
        k = 2 * np.pi / self.wl
        z_R = np.pi * w0**2 / self.wl            # longitud de Rayleigh.
        w = w0 * np.sqrt(1+ (z / z_R)**2)  # radio del haz en z.

        # manejar z=0 sin divisiones por cero
        if z == 0.0:
            R = np.inf
            zeta = 0.0
        else:
            R = z * (1.0 + (z_R / z)**2)
            zeta = np.arctan(z / z_R)

        amp = w0 / w  # factor de amplitud
        envelope = amp * np.exp(- r2 / (w**2))
        phase = np.exp(-1j * (k * z + (k * r2) / (2.0 * R) - zeta))

        E = envelope * phase
        return E
    
    def centerThinLens(self,f,wl,salvar = 'N'):
        """
        Centered phase distribution of a thin lens in paraxial approximation.
        Check Introduction to Fourier Optics, Goodman, 1996, page 99.

        Parameters
        ----------
        f : float
            focal distance of the thin lens
        wl : float
            Coherent light wavelenght
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy

        Returns
        -------
        lens : complex float
            Centered lens phase distribution

        """
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        lens = np.exp(-1j*np.pi/(wl*f)*(X**2+Y**2))
        if salvar == 'Y':
            np.save('tLens_N{}_L{}_wl{}_f{}'.format(self.N,self.L,wl,f),lens)
        return lens
       
    def centerSPP(self,ell,salvar = 'N'):
        """
        Generates a centered spiral phase plate

        Parameters
        ----------
        ell : integer
            Topological charge of the SPP
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy

        Returns
        -------
        spp : complex float
            Centered spiral phase mask

        """
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        theta = np.arctan2(Y,X)
        spp = np.exp(1j*ell*theta)
        if salvar == 'Y':
            np.save('spp_N{}_L{}_ell{}'.format(self.N,self.L,ell),spp)
        return spp

    def centerSquare(self,l,salvar = 'N'):
        """
        Centered square window.

        Parameters
        ----------
        l : float
            side lenght of the square in mm
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy

        Returns
        -------
        back : float
            Centered square window embedded in the working area.

        """
        dx = self.L/self.N
        Dx = int(np.round(l/dx))
        print('--------------------------------------------------------------')
        print('Object: centered square')
        print('Due resolution the true side length is {} mm'.format(Dx*dx))
        print('--------------------------------------------------------------')
        back = np.zeros((self.N,self.N))
        back[int((self.N-Dx)/2):int((self.N-Dx)/2)+Dx,
             int((self.N-Dx)/2):int((self.N-Dx)/2)+Dx] = 1
        if salvar == 'Y':
            np.save('csPlane_N{}_L{}_l{}'.format(self.N,self.L,l),back)
        return back

    def centerWindow(self, u2, Lw):
        """
        Extracts the central part of a matrix

        Parameters
        ----------
        u2 : complex
            Input matrix.
        Lw : integer
            Window side length in mm.

        Returns
        -------
        complex
            New matrix of the window.

        """
        Nw = int(self.N*Lw/self.L)
        return u2[int((self.N-Nw)/2):int((self.N-Nw)/2)+Nw,
                int((self.N-Nw)/2):int((self.N-Nw)/2)+Nw]
    
    def extract_radial_profile(self,image: np.ndarray, center: Tuple[float, float], radius: float, n_angles: int = 360) -> np.ndarray:
        """
        Extrae el perfil de intensidad de una imagen en un círculo de radio fijo desde un punto dado.
        
        Parámetros:
        -----------
        image : np.ndarray
            Imagen 2D en escala de grises.
        center : Tuple[float, float]
            Coordenadas (x0, y0) del centro desde donde medir el radio.
        radius : float
            Radio desde el centro en el que se extraen los valores.
        n_angles : int
            Número de ángulos para muestreo (resolución angular). Por defecto: 360.
    
        Retorna:
        --------
        profile : np.ndarray
            Arreglo 1D de intensidades a lo largo del círculo (longitud = n_angles).
        """
        x0, y0 = center
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        x = x0 + radius * np.cos(thetas)
        y = y0 + radius * np.sin(thetas)
        coords = np.vstack((y, x))  # orden (fila, columna)
        #order=0 para que no elimine los saltos bruscos de interés
        profile = map_coordinates(image, coords, order=0, mode='constant', cval=0.0)
        return profile

    def fork(self, kx, tc = 1, n = 2, salvar='N'):
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        theta = np.arctan2(Y,X)
        forkMask = 0.5*(1 - np.cos(kx*X + 2*np.pi/2 - tc*theta))
        forkMask = np.where((X % (n * 2*np.pi/kx)) < np.pi/kx, 0, forkMask)
        return forkMask    

    def grayLevels(self,u,GL,tipo = 'phase'):
        """
        Discretize the amplitude or phase of a complex object in GL levels.
        It uses floor meaning that it will never reach the maximum value.

        Parameters
        ----------
        u : complex
            Object with amplitude and phase
        GL : integer
            Number of discretization levels
        tipo : string, optional
            The default is 'phase' to discretize the phase of the object.
            'abs' to discretized the amplitude of the object.

        Returns
        -------
        U : complex
            Object with phase or amplitude discretized.
                    if tipo == 'phase':

        """
        if tipo == 'phase':
            nu = np.angle(u) + np.pi
            aux = np.floor(GL*nu/(2*np.pi))
            aux[aux==GL] = GL - 1
            nu = 2*np.pi*aux/(GL)
            U = abs(u)*np.exp(1j*(nu-np.pi))

        elif tipo == 'abs':
            nu = abs(u)
            aux = np.round((GL-1)*nu/np.max(nu))
            nu = np.max(nu)*aux/(GL-1)
            U = nu*np.exp(1j*np.angle(u))
        return U

    def gridBinary(self,a,b,dphi,w,lpm,direc = 'Horizontal',salvar = 'N'):
        """
        Generates a complex binary grating.

        Parameters
        ----------
        a : float
            Amplitude of the step 1
        b : float
            Amplitude of step 2.
        dphi : float
            Phase difference between steps in radians.
        w : float
            Width ratio between steps.
        lpm : integer
            Lines per milimeter
        direc : string, optional
            Direction of the grating. The default is 'Horizontal'.
        salvar : string, optional
            To save as npy use 'Y'. The default is 'N'.

        Returns
        -------
        red : complex
            Diffraction grating

        """
        red = a*np.ones((self.N,self.N))*np.exp(1j*dphi)
        dx = self.L/self.N
        P = 2/lpm
        wPix = w*P/dx
        if (direc == 'Horizontal'):
            barra = b*np.ones((self.N,int(wPix)))*np.exp(1j*0)
            factor = (P/dx-wPix)/(2*wPix)
    
        for ii in np.arange(0,self.N,int(P/dx)):
            posIni = ii + int(wPix*factor)
            if posIni + int(barra.shape[1]) >= self.N: break
            red[:,posIni:posIni+int(barra.shape[1])] = barra
            
        return red
    
    def gridBinaryDefect(self,a,b,dphi,w,lpm,order,direc = 'Horizontal',salvar = 'N'):
        """
        Generates a complex binary grating.
        
        Parameters
        ----------
        a : float
            Amplitude of the step 1
        b : float
            Amplitude of step 2.
        dphi : float
            Phase difference between steps in radians.
        w : float
            Width ratio between steps.
        lpm : integer
            Lines per milimeter
        order : integer
            Order of defect
        direc : string, optional
            Direction of the grating. The default is 'Horizontal'.
        salvar : string, optional
            To save as npy use 'Y'. The default is 'N'.
        
        Returns
        -------
        red : complex
            Diffraction grating
        
        """
        red = a*np.ones((self.N,self.N))*np.exp(1j*dphi)
        dx = self.L/self.N
        P = 2/lpm
        wPix = w*P/dx
        if (direc == 'Horizontal'):
            barra = b*np.ones((self.N,int(wPix)))*np.exp(1j*0)
            factor = (P/dx-wPix)/(2*wPix)
        
        jj = 0
        for ii in np.arange(0,self.N,int(P/dx)):
            jj += 1
            posIni = ii + int(wPix*factor)
            if posIni + int(barra.shape[1]) >= self.N: break
            if jj % order != 0 : red[:,posIni:posIni+int(barra.shape[1])] = barra
            
        return red 
    
    def loadImage(self,name):
        
        # Cargar la imagen con PIL y convertirla en un array de NumPy
        imagen = Image.open(name)
        imagen_array = np.array(imagen)
        #imagen_array = imagen_array[:,:,0]
        
        # Mostrar información de la imagen
        print(f"Forma de la imagen: {imagen_array.shape}")  
        print(f"Tipo de datos: {imagen_array.dtype}")
        
        return imagen_array
    
    def obliqueAstigmatism(self,f):
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y,X)
        Z22 = f*np.sqrt(6)*R**2*np.sin(2*theta)
        return np.exp(1j*Z22)
    
    def shiftRectangle(self, u2, Lv, Lh, shiftV, shiftH):
        """
        Extracts the central part of a matrix

        Parameters
        ----------
        u2 : complex
            Input matrix.
        Lw : integer
            Window side length in mm.

        Returns
        -------
        complex
            New matrix of the window.

        """
        Nv = int(self.N*Lv/self.L)
        Nh = int(self.N*Lh/self.L)
        Nsv = int(self.N*shiftV/self.L)
        Nsh = int(self.N*shiftH/self.L)
        return u2[Nsv + int((self.N-Nv)/2):Nsv + int((self.N-Nv)/2) + Nv,
                Nsh + int((self.N-Nh)/2): Nsh + int((self.N-Nh)/2) + Nh]

    def sphericalAberration(self,f,wl,salvar = 'N'):
        """
        Centered phase distribution of a thin lens in paraxial approximation.
        Check Introduction to Fourier Optics, Goodman, 1996, page 99.

        Parameters
        ----------
        f : float
            focal distance of the thin lens
        wl : float
            Coherent light wavelenght
        salvar : string, optional
            The default is 'N' do not save. 'Y' saves as npy

        Returns
        -------
        lens : complex float
            Centered lens phase distribution

        """
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        lens = np.exp(-1j*np.pi/(wl*f)*(X**2+Y**2)**2)
        if salvar == 'Y':
            np.save('tLens_N{}_L{}_wl{}_f{}'.format(self.N,self.L,wl,f),lens)
        return lens

    def starAiry(self,f,wl,D):
        dx = self.L/self.N
        d = 1.22*wl*f/D
        k = 2*np.pi/wl
        print('--------------------------------------------------------------')
        print('Airy spot from a star')
        print('Central lobe has a sample of {} pixels.'.format(int(d/dx)))
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        R[R==0.0] = 1e-20
        Bessel = 2*np.pi*D*jv(1,k*D*R/f)/(k*R/f)
        u = Bessel*np.exp(1j*k*R**2/(2*f))/(1j*wl*f)
        return u

    def verticalAstigmatism(self,f):
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y,X)
        Z22 = f*np.sqrt(6)*R**2*np.cos(2*theta)
        return np.exp(1j*Z22)

    def verticalTilt(self,wl,p=8,deg=90):
        dx = self.L/self.N
        x = np.arange(-self.L/2,self.L/2,dx)
        X , Y = np.meshgrid(x,x)
        theta = deg*np.pi/180
        alpha = wl/(p*dx)
        return (p*2*np.pi/wl)*(X*np.cos(theta)+Y*np.sin(theta))*np.tan(alpha)