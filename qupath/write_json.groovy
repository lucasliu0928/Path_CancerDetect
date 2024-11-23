def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath('X:/scratch/delete90/haffner_m/user/hrichards/digital_pathology/AimHi')
//mkdirs(pathOutput)

def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
println gson.toJson(annotations)

def outputPath = buildFilePath(pathOutput, name + '_tissue2.json')
File file = new File(outputPath)
file.withWriter('UTF-8') {
    gson.toJson(annotations,it)
}