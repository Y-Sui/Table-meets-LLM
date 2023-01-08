#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import os.path

import requests
import urllib

def read_formid():
    form_jsons = os.listdir("./ms_forms_templates/ms_forms_templates_parsed_20220428") # get formid
    formid = []
    for form_idx in form_jsons:
        with open(os.path.join(f"./ms_forms_templates/ms_forms_templates_parsed_20220428", form_idx), "r") as js:
            form = json.load(js)
            id_code = form["Id"].split("=")[-1].replace("FormId%3D", "")
            id = urllib.parse.unquote(id_code).split("&")[0]  # retrieval the filename (url parsing)
            formid.append(id)
    print(formid)
    return formid

def head_construction(id):
    header = {}
    header['url'] = f'https://forms.office.com/formapi/DownloadExcelFile.ashx?formid={id}&timezoneOffset=-480&__TimezoneId=Asia/Shanghai&minResponseId=1&maxResponseId=31'
    header['headers'] = {'Host': 'forms.office.com', 'Connection': 'keep-alive', 'sec-ch-ua': '"Microsoft Edge";v="107", "Chromium";v="107", "Not=A?Brand";v="24"', 'odata-version': '4.0', 'x-correlationid': '6eed9642-a726-4fa6-8d41-2839c924645d', 'x-ms-form-muid': '3B122A9488556A112F0B38F489166B56', 'x-usersessionid': 'f572bcb9-134e-48f2-b99a-6279ad40c386', 'x-ms-form-request-ring': 'business', 'sec-ch-ua-mobile': '?0'}
    header['cookies'] = {'MSFPC': 'GUID', 'FormsWebSessionId': '486aa9b0-6633-4c99-8f7c-382ccfd67431', 'usenewauthrollout': 'True', 'RpsAuthNonce': '39ddb77e-acd8-4c1f-9220-9655b667af83', 'MUID': '3B122A9488556A112F0B38F489166B56', '__RequestVerificationToken': 'TjGDXuifHHrYghjMKKu6YhDf7smbnPXIF6GDkNEpOT__S3WO-D_eKqj0farJcEWNSSaYVd7JnoyVjwbtyURHEDssq7rNjKsei9E7FZMZec41', 'OhpAuthToken': 'eyJ0eXAiOiJKV1QiLCJub25jZSI6InFZbkx1SFMyZjZaZWhKUy1mS1BrT0lGMDctUUdoWFZaUGt2NHR6b2lhYTgiLCJhbGciOiJSUzI1NiIsIng1dCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSIsImtpZCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSJ9.eyJhdWQiOiI0MzQ1YTdiOS05YTYzLTQ5MTAtYTQyNi0zNTM2MzIwMWQ1MDMiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNjY5ODI3MDA2LCJuYmYiOjE2Njk4MjcwMDYsImV4cCI6MTY2OTgzMTk3OCwiYWNyIjoiMSIsImFpbyI6IkFWUUFxLzhUQUFBQTZCUnROU096S3ptek5YUGZNSktKWWQ2RVVZWFB6QnVrRlRYWVZIZlJLbWd1aWpYRm9xaEhIVWF6dEl3QUk2bTEyZlExS2s3dU9MTm1TK2wxY3dTNWw1dG1CQ3VxRWl2YjkwUno3NnFkc0tVPSIsImFtciI6WyJwd2QiLCJyc2EiLCJtZmEiXSwiYXBwaWQiOiIwZWM4OTNlMC01Nzg1LTRkZTYtOTlkYS00ZWQxMjRlNTI5NmMiLCJhcHBpZGFjciI6IjAiLCJkZXZpY2VpZCI6IjlkZGQ5MWJkLTIyZWEtNGYwYS1iYjJhLWVjMjU4NWFhNGNiYyIsImZhbWlseV9uYW1lIjoiU3VpIiwiZ2l2ZW5fbmFtZSI6Ill1YW4iLCJpcGFkZHIiOiIxNjcuMjIwLjIzMi4xMyIsIm5hbWUiOiJZdWFuIFN1aSAoRkEgVGFsZW50KSIsIm9pZCI6IjI1OThhNGZkLTQ5ZmUtNDg0OC04NWFkLWQ1N2ZhNjBkZDE0MiIsIm9ucHJlbV9zaWQiOiJTLTEtNS0yMS0yMTQ2NzczMDg1LTkwMzM2MzI4NS03MTkzNDQ3MDctMjg1Mjc1NiIsInB1aWQiOiIxMDAzMjAwMjE4MkI5RDI3IiwicmgiOiIwLkFSb0F2NGo1Y3ZHR3IwR1JxeTE4MEJIYlI3bW5SVU5qbWhCSnBDWTFOaklCMVFNYUFDOC4iLCJzY3AiOiJPZmZpY2VIb21lLkFsbCIsInNpZCI6Ijg5ZTU0OWU4LTIwZGUtNGExZi05N2M2LTk5NDFjOTY2NTNlZSIsInN1YiI6IllEdFZpdEJ0b3RveUpEcGhDdUpKLWIzSFdPX29FM3p0M2tuWFNJZHhQUEkiLCJ0aWQiOiI3MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDciLCJ1bmlxdWVfbmFtZSI6InYteXVhbnN1aUBtaWNyb3NvZnQuY29tIiwidXBuIjoidi15dWFuc3VpQG1pY3Jvc29mdC5jb20iLCJ1dGkiOiJTaHhhdWpHTUdVQ29VTXlUN1M2YUFRIiwidmVyIjoiMS4wIn0.aVG5yTYVvY7vX-THDGIXBn3-AEc4PQMKza1AGhtKijeE7izHEu544aZQKRAIUmQ3vvoxF0cDdJgm6_2YaIFl33i9M7LB2Dxlxn6clmwSr4NOEMD8pZfjw9GWwlOSOqFp42ChZF32cR7Loz3haNi_kSgHYN71K2pLacLJVdamUoJqrsaxaqqPqOPjVCy9_C5cDt3nPJQ9F7wpNNH5In8A89fVkZW6lFNeEAvf_kJM-P1garORJu7GIkzjufPHCVlPLK5oZGUza6rGqOHnxgp0RIs737-zW_8Xsx--9oI4JOvcOoHeXJx736DnupB0Bb_fzq5WNtTKGrFIBA9-rBD_dw', 'AADAuth.forms': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSIsImtpZCI6IjJaUXBKM1VwYmpBWVhZR2FYRUpsOGxWMFRPSSJ9.eyJhdWQiOiJjOWE1NTlkMi03YWFiLTRmMTMtYTZlZC1lN2U5YzUyYWVjODciLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNjY5ODI4ODM0LCJuYmYiOjE2Njk4Mjg4MzQsImV4cCI6MTY2OTgzMjczNCwiYWlvIjoiQVdRQW0vOFRBQUFBYUhlWUp1a1pkSkRCUEN1eDU1UFZyZjErSGJySkZrMnBCUmtyZ1kxTFVUbTJvcDVDS2pzNW9kbUNTTGtDRU5EVlR1MHpFSkFaMFhDN1VjM2FHN3lidEgyMzFZWU02RzV0bDJZQ2dkR1d1eVZkdWM4TzhEcG5YYVI0elBmdTZlN0QiLCJhbXIiOlsicHdkIiwicnNhIiwibWZhIl0sImNfaGFzaCI6InVrRDdiMVNMb0ZRNlBIM0pJVjg5M3ciLCJmYW1pbHlfbmFtZSI6IlN1aSIsImdpdmVuX25hbWUiOiJZdWFuIiwiaXBhZGRyIjoiMTY3LjIyMC4yMzMuMTMiLCJuYW1lIjoiWXVhbiBTdWkgKEZBIFRhbGVudCkiLCJub25jZSI6IjYzODA1NDI1OTM0NjI2MzY4Ny5ZalJtTURVek1EVXRNbVpsT1MwMFpESTNMVGs0Wm1NdE9ETmlaRFJqT0daak5qQXhZakUzTUdJMU5UTXRaV0UzTVMwME56VmlMVGhrWldZdE1XUmpOV1F5TmpWbFpEQTQiLCJvaWQiOiIyNTk4YTRmZC00OWZlLTQ4NDgtODVhZC1kNTdmYTYwZGQxNDIiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMjE0Njc3MzA4NS05MDMzNjMyODUtNzE5MzQ0NzA3LTI4NTI3NTYiLCJwdWlkIjoiMTAwMzIwMDIxODJCOUQyNyIsInJoIjoiMC5BUm9BdjRqNWN2R0dyMEdScXkxODBCSGJSOUpacGNtcmVoTlBwdTNuNmNVcTdJY2FBQzguIiwic3ViIjoiazkzTDlidmtFQUFuc3BQbHNGOEU5RDFaVWZjY2ZvMEltcDdmNUVUTGcxayIsInRpZCI6IjcyZjk4OGJmLTg2ZjEtNDFhZi05MWFiLTJkN2NkMDExZGI0NyIsInVuaXF1ZV9uYW1lIjoidi15dWFuc3VpQG1pY3Jvc29mdC5jb20iLCJ1cG4iOiJ2LXl1YW5zdWlAbWljcm9zb2Z0LmNvbSIsInV0aSI6IlQza0pyZnlfYkVhb2RQSzZDX2FmQWciLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbImI3OWZiZjRkLTNlZjktNDY4OS04MTQzLTc2YjE5NGU4NTUwOSJdfQ.peDELTGCw0rSwHe50pTdPAGDYsKbCMV8A35Wo9JcQC9NuAb9cWHb2fnTmljbuVNLcwJW23xEybUf0eNC_kLSCEFtQ1b97laks9ytnAwN6CAyAgjN4TU_K2mVo87SdlYwkrM6nKx6U4LKbTu44xaPidbPcJm3JlO1EdNFdl8y8-brwzuy8JWb6CTg6wzeTvAtayDDAKzpzgWJ8Gw9fu-kkgO01PiNKlWdwVOmM75QLuFYlQVZkKX88_PjvyCWRm2J1MzmMs_8IRLThi2bhrvyyNo-Es5gCZ8ZTqjtY3DIzYlsaYNogUkCFeLGJ1b5pF0WC1wOVHNcJbpd0ZBD0jHcUQ', 'AADAuthCode.forms': '0.ARoAv4j5cvGGr0GRqy180BHbR9JZpcmrehNPpu3n6cUq7IcaAC8.AgABAAIAAAD--DLA3VO7QrddgJg7WevrAgDs_wQA9P_51DRQcH_kUOrthZ40psdfPVTUbPmL7y4nMQ3A3Rzl8nZZP5nZL6xDGFITeU0-jLoZB_cNisepAb6GYYfy3od0SGIzQlqVlwUiud4htGxlWEaaJYFuaCgQkC96k54tbWBhU5edpRUd0b906WIH2bvsljr-e3Ldjl8Glp_q87QiKMp7tErS8iH9fv-Exnz-_YvDiPMuE8NzHbCRq_ztbWC5mgUn52kXmhBD2RleU-wWZoJtPe9qKfbUKx7sB-HHh0nfKC_iTKV2baFBQ7VOaRZqze51eUAtmyr-kbR0CzHbrgSNoofP0YbQkJjjsPsN2cD660_cPC6nSKv2Mz3PDowx42AEjnygbhHx6VWCyPVDe1X_urZCHR7tUHAi4RufjqPjqD2yXi4TYxAWfEBI35eIXuX2NFMrNGP-fdN4JRlpBd71ldUucohe_fWyWfCmOGNfipwPm9uLr8EW-QJkq7-LN6gw2uLaV0nlfByb5dksTdtLyv5dmL40GiwU-nP233-s9Bf4BfBD_vAyMilf_gTQ9-gzrI2-BJMLzYFy7IXB0XExkqT-0KFZ6Ex_CVPKViXSnvqJxAN7Gx88_LWpDfCte6ihMqdcw3A2rGuo0LX47yo00pnLqy_RMlXbFxXsUuvXjUh0LNlJto3vAiWKObV2NO_2ylyK-PO6ENWy2Bxy6lQMdektKypztOFvzFhegupABeUFaMzlJgiw4D-zgLsJzxP--T4nYL0wSRC85JX8sZ3L3e4T--XTk8ITb4nXWl6-hecr2fbIAgxkmrbKE6Sq_lIyNOsYczIWJmmGFxRis6YC977RPk_wMFqXwSxmaMpRoZvIuB8_NF7wVdrP4_Dma7tEmbfqpl9EaT1QAoVDzlp-5Ho7nHMjZpC46-lu7icKLgJI-Qz13X-2gk2ovebKjtwX00muNOq5ZMKyXN6_MDsZ5S6Nng_kiARwpabqNwGYJEp_s7qP8S7NwvWIgCB3KMs1r3MsXZMSDCFQKdnCduBYFTEiTVbO6PJb', 'OIDCAuth.forms': 'Aa6GwyqHqLEgns8j-IRfJFdGaNNrAl9z_xIXLKFrF-k8TjR6PWGU1FzZfPNxRYdE3vcGfKFbQ3UyQhGAKBCx81JlyM-ev5BFh_MuBOATlZ9LhqyoIZUFDaO7mL2CyEmFQt3jAP39HuWqPfwxDwmsDrr5WcmA8f7ZN4t1hfuRfysUlFZvcBVF4Xz6zOJrH-OFEl1SoEh3uVarAi8je2RrpgGlJDC41SVi9v84Rp3JK3kElM8k3c-2T8hYhwzWHW0ZAmHt0K5WJAojBfRl3NI_4f0polp6L3AUfVbKb7waNRUG2fH3QPXRkG03mqL6FiO5P8bi0J4GYiH9kK3Nlycs6AEDc8YBOYtPN_krfOe8pMq7R9pGa38uCEVbmkjRaRcoDPhGDyL8IoRwwMKhy9FwA2mCuqaGlAM7C2Aw4RwOu3ejE0TJxR4G9Dg6IePa4lc0fTyajQfzJr0Uk6wtT4V58yWj46-9MYnaM41M1BiCW5Ta3iGEByRjRgQLCp2-rHpAZJjNa2vPr3Rx8WW70JVAQcVboHrXk78qXTf7exmuHQwWVeAc-EokvBvB-cds6LvQjvRdNtjvea6GRdLrEeT-82gCzHqqaKGX67Psmfnl42BZKPta-0NsmN4dq_7WI89fGyyfkwD39pUrtfK4J4OszQqm1Kgmeaf08szBITSkZuwZlgX13t7X9XibGK4H9IuFydmZ6RlGagN89clfi0asaWU4FdPW6P2NNeADs09Kp1ZlFoBH6Mraij-vc0Nj5279rlLp1vdvJkYsqg9lFRjY6SxqCzN0db2htkRNTfvx-tmcUHgvHGmf5plkNh_n1x79IK_1KzubiBwjFWI4PB3nm5xOoDfKXSd_iZmThVsENcSRgxxndA4z9i6YxhtNhra46yhPfajY7tJRjznjA0YBtIrtpMHPulYAAC-VgQssiIomtj-LA528v7o5eiIAYFj_GYsrRA0050aZnYnrhbIRxaRQhBAuDLz8twItsRHRqEsh96r8Ebia6hBgnUrwuqfOYDFQZVbxFRwk3lTSjPUO7fBCdH9HThP68VobIm3WbJhgLouRxsbODh5_XFGot5UqfB6IvTjjRMR14-6McTHTgQ5cpPK4hd_Kurt6oRhRkuXyhSkVhLsrQEwzkP-h_vinY-5YX7K0ZFq_L8u9A2JzXCno5h2Ysh020AlZRW1CC0ONiTNy3lyjfuablYojLg5QEvo2a_viBrSC5OFNuJjbqG6Pe58i6j38jK1KKDGzL06KQGtCm-Hx4R83YqBS2L_N3pic0LZETwHt0e9Hcivj9dm_XG5RhQEyfaLLzmfwn4pRw7OVwaXjfm1J_4hTPSVE6HnFM0Ux9yLIvZYw6mY_8MsgXgkrV5_nAs0_SlsPvZz09wliezuGdX07Hau1xOQFaAXchjhja27qTHviyFe2PpzmWGjOL-wsEjMJaFOt5yhOG2QP-lB5n-SwISiUO6pKCNTptp4OmONzV2nfdc4MDyniOg0CSvssoxjbwYbDuEM3QmIZU-_rEuIFPkH-QN5GZ8Ubm3qfAXZIJC7nLztWfm3utz-ksKIREixZTft6zNoiCimIHOb00WdizqpynE7E8BkjYT6i5irFFK6TZRtPS2bLTKP28bNiVjG56FCTaWG-jeko1m0OY0ImcMsvSibxTwMLUKOQiBpdQsfcb7HnpeJToa-2yfSG3O6-_FROWo66_Q_Gc-K2Q6X5hfGo8IOxmlJZNgik-Y_g19c5JNr2jJES64H8KcVvcRAPGiJgwF6jHpVIQLD3-6zwHZBXgECoKA', 'MicrosoftApplicationsTelemetryDeviceId': 'dbb4121c-48c4-4c4f-a53e-fbb647bae2f6', 'ai_session': '04bL/b0ZMYQRoI8pHe3ldd|1669831875291|1669831875291'}
    return header

def main():
    formid = read_formid()
    num = 0
    for id in formid:
        num += 1
        header = head_construction(id)
        response = requests.get(header['url'], headers=header['headers'], verify=False, cookies=header['cookies'])

        print(response)

        if response.status_code == 200:
            # filename_code = response.headers.get("content-disposition").split("=")[-1].replace("UTF-8''", "")
            # filename = urllib.parse.unquote(filename_code)  # retrieval the filename (url parsing)
            filename = f"ms_forms_response_{num}.xlsx"
            filepath = os.path.join("./generated/ms_forms_response", filename)
            if response.text:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1):
                        f.write(chunk)
            else:
                print("the file is empty")
        else:
            print("credential has expired")

if __name__ == "__main__":
    main()