
#!/usr/bin/python3
from flask import Flask, request, render_template, send_file, abort
from io import BytesIO 

# Our own modules
import searchService
import imageService

app = Flask(__name__)

# These modules will add a http route '/' for example and assosiate a return value from the function 

# first page /
@app.route('/', methods=['GET'])
def index():
  return "Hello Koder. This is Towyse Debug ubuntu service with flask app.<br> Go to <a href='/home'>home</a>"

# test 
@app.route("/home")
def home():
    return render_template('index.html', title="HOME", message="You are home"  )

# test 
@app.route('/hello', methods=['GET'])
def hello():
    return render_template('index.html', title="Info", message="Hello. How are you? /hello"  )



# ------------------- SERVERS ------------------#

# Generic Image Service API with images from local drive
@app.route('/api/image/<string:image_id>', methods=['GET'])
def get_image(image_id):
    img_io = BytesIO()
    pil_img = imageService.get_image_by_id_from_local(image_id)   

    if pil_img ==0 : 
      abort(404) 
    
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)    
    return send_file(img_io, mimetype='image/jpeg')




# test for images from API Service;  endpoint: /images 
@app.route('/images_api_test', methods=['GET'])
# test: display a list of images
def get_testimages():
    # from API service 
    imgs = [ 
            { "link": "/api/image/1" , 'text' : '20' }, 
            { "link": "/api/image/2" , 'text' : '20' }, 
            { "link": "/api/image/3" , 'text' : '20' }, 
            { "link": "/api/image/4" , 'text' : '20' } 
    ]

    return render_template('images.html', images=imgs  )




# test for images from /static directory; endpoint: /static
@app.route('/images_static_test', methods=['GET'])
# test: display a list of images
def get_teststaticimages():

    #  from static directory and from api service 
    imgs = [ 
            { "link": "/static/images/1.jpg" , 'text' : '20' }, 
            { "link": "/static/images/2.jpg" , 'text' : '20' }, 
            { "link": "/static/images/3.jpg" , 'text' : '20' }, 
            { "link": "/static/images/4.jpg" , 'text' : '20' } 
    ]

    return render_template('images.html', images=imgs  )



# search images template test 

@app.route('/searchimages', methods=['GET'])
@app.route('/searchimagestest', methods=['GET'])

# example display a query image and list of searched images
def search_images():
    qimg = {"link": "/static/images/7.jpg" , 'text' : 'Input Image!'}
    imgs = [ 
            { "link": "/static/images/1.jpg" , 'text' : 'Score: 20' }, 
            { "link": "/static/images/2.jpg" , 'text' : 'Score: 20' }, 
            { "link": "/static/images/3.jpg" , 'text' : 'Score: 20' }, 
            { "link": "/static/images/4.jpg" , 'text' : 'Score: 20' } 
    ]
    return render_template('search_images.html', images=imgs, qimage = qimg )



#  API query with Image ID string to search from database
@app.route('/searchimagestest/<string:image_id>', methods=['GET'])
def search_imagestest(image_id):
      
    # this is a test
    qimg = {"link": "/static/images/"+image_id+".jpg" , 'text' : 'Input Image!'}
    imgs = searchService.imagelist()  # alternative from static dir 
    print (imgs)
    return render_template('search_images.html', images=imgs, qimage = qimg )


#  Displays a list of sampled images (default=3) for database 
@app.route('/images', methods=['GET'])
# test: display a list of images
def get_sampleimages():
      
    imglist = imageService.image_sample_ID(4)
    imgs = mapID(imglist)
    return render_template('images.html', images=imgs  )


#  API query with Image ID as string as in database
@app.route('/searchimages/<string:image_id>', methods=['GET'])
def search_image(image_id):
      
    qimg = {"link": "/api/image/"+image_id , 'text' : 'Input Image!'}

    # actual implementation of the algo 
    imglist = searchService.imageIDList(image_id)
    imgs = mapID(imglist)

    return render_template('search_images.html', images=imgs, qimage = qimg )


# Helper Function: Map list of ids to /api/image/
# input dict  : {'id': 'value', text: 'textvalue'}
# output dict : {'link', '/api/image/value', 'text': 'textvalue', 'search': 'ID'}
def mapID(idList):   
    imgs = []
    API_ROOT = "/api/image/"
    for img in idList : 
      link = API_ROOT + str(img['ID'])
      print (link)
      imgs.append({'link': link, 'text': img['text'], 'search': '/searchimages/'+ str(img['ID'])})
    return imgs


# ----------------------------- END ----------------------------- #

@app.route('/bye', methods=['GET'])
def bye():
    return "Bye now."


# -----------------------SERVER HOST and PORT -------------------- #

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=9006)