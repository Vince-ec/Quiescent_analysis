import numpy as np
import Constants as C
from Get_sensitivity import Get_Sensitivity
from scipy.interpolate import interp1d

def Sig_int(nu,er,trans,energy):
    sig = np.zeros(len(nu)-1)
    
    for i in range(len(nu)-1):
        sig[i] = (nu[i+1] - nu[i])/2 *np.sqrt(er[i]**2 * energy[i]**2 * trans[i]**2 + er[i+1]**2 * energy[i+1]**2 * trans[i+1]**2)
    
    return np.sum(sig) / np.trapz(trans * energy, nu)

class Photometry(object):

    def __init__(self,wv,fl,er,filter_number):
        self.wv = wv
        self.fl = fl
        self.er = er
        self.filter_number = filter_number

    def Get_Sensitivity(self, filter_num = 0):
        if filter_num != 0:
            self.filter_number = filter_num

        f = open('vtl/FILTER.RES.latest', 'r')
        data = f.readlines()
        rows = []
        for i in range(len(data)):
            rows.append(data[i].split())
        i = 0
        sens_data = []
        while i < len(data):
            sdata = []
            amount = int(rows[i][0])
            for u in range(amount):
                r = np.array(rows[i + u + 1])
                sdata.append(r.astype(np.float))
            sens_data.append(sdata)
            i = i + amount + 1

        sens_wave = []
        sens_func = []
        s_wave = []
        s_func = []
        for i in range(len(sens_data[self.filter_number - 1])):
            s_wave.append(sens_data[self.filter_number - 1][i][1])
            s_func.append(sens_data[self.filter_number - 1][i][2])

        for i in range(len(s_func)):
            if .001 < s_func[i]:
                sens_func.append(s_func[i])
                sens_wave.append(s_wave[i])

        self.sens_wv = np.array(sens_wave)
        self.trans = np.array(sens_func)

    def Photo(self):
        h = C.h  # planck constant erg s
        c = C.c  # speed of light cm s^-1
        atocm = C.angtocm

        wave = self.wv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans,energy) * (c / (wave_eff * atocm) ** 2)

    def Photo_clipped(self):

        IDX = [U for U in range(len(self.sens_wv)) if self.wv[0] < self.sens_wv[U] < self.wv[-1]]

        h = C.h  # planck constant erg s
        c = C.c  # speed of light cm s^-1
        atocm = C.angtocm

        wave = self.wv * atocm
        filtnu = c /(self.sens_wv[IDX] * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans[IDX]
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans[IDX] * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        top1 = Ernu * energy * self.trans[IDX]
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans[IDX] * energy
        bottom = np.trapz(bottom1, filtnu)
        erphotonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)
        photoer = erphotonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans[IDX],energy) * (c / (wave_eff * atocm) ** 2)
        
    def Photo_model(self,mwv,mfl):
        h=C.h # planck constant erg s
        c=C.c          # speed of light cm s^-1
        atocm=C.angtocm

        wave = mwv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * mfl
        Fnu = interp1d(nu, fnu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)
        photo = photonu * (c / (wave_eff * atocm) ** 2)
        
        self.eff_mwv = wave_eff
        self.mphoto = photo

    def FWHM(self):
        top = np.trapz((self.trans * np.log(self.sens_wv/self.eff_wv)**2) / self.sens_wv, self.sens_wv)
        bot = np.trapz(self.trans / self.sens_wv, self.sens_wv)
        sigma = np.sqrt(top/bot)

        self.fwhm = np.sqrt(8*np.log(2))*sigma * self.eff_wv