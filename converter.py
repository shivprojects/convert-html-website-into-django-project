import os,re,shutil,sys,getopt

from subprocess import PIPE, Popen

class HtmlToDjango():
    def __init__(self,project_name, workdir,html_website, django_version):
        self.run_dir = os.path.join(workdir,"project",project_name)

        self.build_path  = os.path.join(workdir,"builds",django_version,"mysite")
        self.convert_path  = os.path.join(self.run_dir,"user-web")
        self.project_path = os.path.join(self.run_dir,"mysite")
        self.virtual_env = os.path.join(self.run_dir,"virtual-env")
        self.python_version = str(sys.version_info.major)
        self.project_name = project_name 
        self.html_website = html_website
        self.django_version = django_version

        self.index_page = ""
        self.p404_page = ""
        self.files = []
        self.dirs = []
        self.extras = []

    def make_project(self):

        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir) 

        os.makedirs(self.run_dir)

        cmd = "\
        cd %s;\n\
        virtualenv --python=python%s %s;\n\
        source %s/bin/activate;\n\
        pip install django==%s ;\n\
        mkdir %s ;\n \
        cd %s ;\n \
        django-admin startproject mysite .;\n\
        python%s manage.py startapp %s ;\n\
        cd -\n\
        " %(self.run_dir, self.python_version, self.virtual_env,self.virtual_env,self.django_version, self.project_path, self.project_path, self.python_version, self.project_name)
        
        f = open(os.path.join(self.run_dir,"make_project.bash"),"w")
        f.write(cmd)
        f.close()
        print ("INFO : Creating virtual Environment & installing django %s" %(self.django_version))
        command = ("bash %s/make_project.bash") %(self.run_dir)
        out = Popen(args=command,stdout=PIPE,shell=True).communicate()[0]
        shutil.copytree(self.html_website,self.convert_path)

    def convert(self):
        print ("INFO : Converting user setup ...")
        for content in os.listdir(self.convert_path):
            dir_path = os.path.join(self.convert_path, content)
            if os.path.isdir(dir_path):
                self.dirs.append(content)
            elif os.path.isfile(dir_path) and re.search(".html$",content):
                m = re.search("(\S+).html$",content)
                if m:
                    name = m.group(1)
                    self.files.append(name)

                    if re.search("404",name):
                        self.p404_page = name
                    elif self.index_page:
                        if re.search("index",self.index_page):
                            continue
                        elif re.search("index|home|main",name):
                            self.index_page = name
                    elif not self.index_page and re.search("index|home|main",name):
                        self.index_page = name
            else:
                self.extras.append(content)
                
        for content in self.files:
            path = os.path.join(self.convert_path, content+'.html')
            f = open(path, "r")
            lines = "".join(f.readlines())
            f.close()
            for dir in self.dirs:
                lines = lines.replace(dir+'/','/static/'+self.project_name+'/'+dir+'/')
            for file in self.extras:
                lines = lines.replace(file,'/static/'+self.project_name+'/extra/'+file)
            for file in self.files:
                lines = lines.replace(file+'.html',file)
            f = open(path, "w")
            f.write(lines)
            f.close()

    def map_files(self):
        print ("INFO : Mapping files with django ...")
        template_path = os.path.join(self.project_path,self.project_name,"templates",self.project_name)
        static_path = os.path.join(self.project_path,self.project_name,"static",self.project_name)
        helperScripts = os.path.join(os.getcwd(),"scripts")

        if not os.path.exists(template_path):
            os.makedirs(template_path)

        if not os.path.exists(static_path):
            os.makedirs(static_path)

        if len(self.extras) > 0:
            os.makedirs(os.path.join(static_path,"extra"))

        for file in self.extras:
            path = os.path.join(self.convert_path,file)
            extra_path = os.path.join(static_path,"extra")
            shutil.copy2(path,extra_path)

        for file in self.files:
            file_name = file + '.html'
            file_path = os.path.join(self.convert_path,file_name)
            shutil.copy2(file_path,template_path)

        for dir in self.dirs:
            file_path = os.path.join(self.convert_path,dir)
            static_dir = os.path.join(static_path,dir)
            shutil.copytree(file_path, static_dir)

        # --------- mysite settings.py file --------------#
        setting_file = os.path.join(self.project_path,"mysite/settings.py")
        f = open(setting_file,"r")
        lines = []
        for line in f:
            lines.append(line)
            if re.search("^\s*INSTALLED_APPS",line):
                lines.append("    '%s',\n" %(self.project_name))
        f.close()
        f = open(setting_file, "w")
        f.write("".join(lines))
        f.write("STATIC_ROOT = u'/var/www/static'\n")
        f.close()

        # --------- mysite urls.py file --------------#
        urls_file = os.path.join(self.project_path,"mysite/urls.py")
        f = open(urls_file,"r")
        lines = []
        for line in f:
            lines.append(line)
            if re.search("\^admin\/",line):
                lines.append("    url(r'', include('%s.urls'))," %(self.project_name))
                lines.append("    url(r'.*', include('%s.urls'))," %(self.project_name))
        f.close()
        f = open(urls_file, "w")
        f.write("from django.conf.urls import include\n")
        f.write("".join(lines))
        f.close()

        # --------- project urls.py file --------------#
        urls_file = os.path.join(self.project_path,self.project_name,"urls.py")
        f = open(os.path.join(helperScripts,"urls.py"),"r")
        lines = "".join(f.readlines())
        f.close()

        import_name = "."
        if re.search("^2",self.python_version):
            import_name = self.project_name
        lines = lines.replace("IMPORT_VIEWS","from %s import views" %(import_name))
        lines = lines.replace("PROJECT_NAME",self.project_name)

        f = open(urls_file,"w")
        f.write(lines)
        f.close()

        # --------- project views file --------------#
        views_file = os.path.join(self.project_path,self.project_name,"views.py")
        f = open(os.path.join(helperScripts,"views.py"),"r")
        lines = "".join(f.readlines())
        f.close()
        lines = lines.replace("PROJECT_NAME",self.project_name)

        if not self.p404_page:
            shutil.copy2(os.path.join(helperScripts,"404-page.html"),template_path)
            self.p404_page = "404-page"
        lines = lines.replace("404_PAGE",self.project_name+"/"+self.p404_page+'.html')

        if not self.index_page:
            shutil.copy2(os.path.join(helperScripts,"index.html"),template_path)
            self.index_page = "index"
        lines = lines.replace("INDEX_PAGE",self.project_name+"/"+self.index_page+'.html')

        f = open(views_file ,"w")
        f.write(lines)
        f.close()
        
        # Removing convert path
        shutil.rmtree(self.convert_path)
        print ("INFO : See project at path : %s " %(self.run_dir ))


if __name__ == "__main__":
    #--------------------------------------------#
    workdir = os.getcwd()
    html_website = ""
    project_name = ""

    django_version = "1.9.0"

    try:
        html_website   = sys.argv[1]
        project_name  = sys.argv[2]
        if len(sys.argv) == 4:
            django_version = sys.argv[3]
    except:
        print ("%s <html files path> <website name> <django version default 1.9.0>" %(sys.argv[0]))
        sys.exit()

#--------------------------------------------#
    print ("INFO : Started ...")
    convert_obj = HtmlToDjango(project_name, workdir,html_website, django_version)
    convert_obj.make_project()
    convert_obj.convert()
    convert_obj.map_files()
    print ("INFO : Finished !!!")
