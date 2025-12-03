// 한국 전통 탈 정보 데이터베이스
export interface MaskDetailInfo {
  name: string;
  koreanName: string;
  origin: string;
  description: string;
  danceRole: string;
  quote: string;
  wikiUrl?: string;
}

export const MASK_INFO: Record<string, MaskDetailInfo> = {
  // 하회탈 (안동 하회마을)
  yangban: {
    name: 'yangban',
    koreanName: '양반',
    origin: '안동 하회탈',
    description: '양반 계층을 풍자하는 탈로, 턱이 분리되어 대사에 따라 움직입니다. 권위적이면서도 우스꽝스러운 표정이 특징입니다.',
    danceRole: '선비를 데리고 다니며 백정에게 우롱당하는 위선적 양반 역할',
    quote: '이놈, 상놈이 감히 양반을 희롱하느냐!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  bune: {
    name: 'bune',
    koreanName: '부네',
    origin: '안동 하회탈',
    description: '젊은 여인을 상징하는 탈로, 아름다운 미소와 부드러운 곡선이 특징입니다. 한국 전통 탈 중 가장 아름다운 여성상으로 평가받습니다.',
    danceRole: '중의 유혹을 받으며 양반과 선비가 다투게 만드는 젊은 여인 역할',
    quote: '아이고, 저를 두고 싸우시다니요...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  gaksi: {
    name: 'gaksi',
    koreanName: '각시',
    origin: '안동 하회탈',
    description: '신부 또는 새색시를 상징하는 탈입니다. 부네보다 더 어리고 수줍은 표정을 하고 있습니다.',
    danceRole: '혼례 장면에서 순결하고 정숙한 새색시 역할',
    quote: '부끄러워라, 고개를 들 수가 없구나...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  jung: {
    name: 'jung',
    koreanName: '중',
    origin: '안동 하회탈',
    description: '파계승을 상징하는 탈로, 탐욕스럽고 음탕한 표정이 특징입니다. 불교 승려의 타락상을 풍자합니다.',
    danceRole: '부네를 유혹하는 위선적인 파계승 역할',
    quote: '나무아미타불... 아니, 저 여인의 자태란!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  imae: {
    name: 'imae',
    koreanName: '이매',
    origin: '안동 하회탈',
    description: '바보 또는 천치를 상징하는 탈로, 한쪽 눈이 찌그러지고 입이 비뚤어진 비대칭 얼굴이 특징입니다.',
    danceRole: '어리숙하지만 양반의 허세를 본의 아니게 폭로하는 역할',
    quote: '에헤헤, 그게 뭔 소리여?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  chorangi: {
    name: 'chorangi',
    koreanName: '초랭이',
    origin: '안동 하회탈',
    description: '양반의 하인을 상징하는 탈로, 재치 있고 영리한 표정이 특징입니다. 주인을 놀리면서도 충직한 하인의 모습을 보여줍니다.',
    danceRole: '양반을 수행하며 그의 허세를 꼬집는 영리한 하인 역할',
    quote: '나으리, 그건 좀 아닌 것 같은디요...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  baekjeong: {
    name: 'baekjeong',
    koreanName: '백정',
    origin: '안동 하회탈',
    description: '도살업자를 상징하는 탈로, 투박하고 거친 인상이 특징입니다. 천민 계층이지만 실제 능력이 있는 인물을 표현합니다.',
    danceRole: '소의 생식기를 들고 양반을 조롱하는 통쾌한 역할',
    quote: '양반 나으리, 이것 좀 보시오!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  halmi: {
    name: 'halmi',
    koreanName: '할미',
    origin: '안동 하회탈',
    description: '늙은 여인을 상징하는 탈로, 주름진 얼굴과 이빨 빠진 입이 특징입니다. 고단한 삶을 살아온 서민 여성을 표현합니다.',
    danceRole: '억척스럽게 살아온 서민 여성의 애환을 표현하는 역할',
    quote: '이 늙은 것이 평생을 고생만 했구나...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  
  // 봉산탈 (황해도 봉산)
  malttugi: {
    name: 'malttugi',
    koreanName: '말뚝이',
    origin: '봉산탈춤',
    description: '양반의 하인으로, 봉산탈춤에서 가장 인기 있는 캐릭터입니다. 해학적이고 풍자적인 대사로 양반을 조롱합니다.',
    danceRole: '채찍을 들고 양반 삼형제를 신랄하게 비판하는 하이라이트 역할',
    quote: '쉬이~ 양반 나오신다! 양반이라고 하니깐 노론, 소론, 호조, 병조...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  miyal: {
    name: 'miyal',
    koreanName: '미얄',
    origin: '봉산탈춤',
    description: '영감의 본처로, 남편에게 버림받은 늙은 여인입니다. 서글프면서도 익살스러운 표정이 특징입니다.',
    danceRole: '영감과 첩 덜머리집 사이에서 쫓겨나는 비극적 역할',
    quote: '이 늙은 년을 버리고 젊은 년을 취하다니...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  chwibari: {
    name: 'chwibari',
    koreanName: '취발이',
    origin: '봉산탈춤',
    description: '파계승을 상징하는 탈로, 머리카락이 헝클어진 모습이 특징입니다. 술에 취한 듯한 붉은 얼굴이 인상적입니다.',
    danceRole: '소무를 유혹하는 술 취한 파계승 역할',
    quote: '아이고 취했다, 저 각시 좀 보소!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  nojang: {
    name: 'nojang',
    koreanName: '노장',
    origin: '봉산탈춤',
    description: '늙은 고승을 상징하는 탈입니다. 도를 닦은 듯 하지만 결국 세속적 욕망에 빠지는 위선적 모습을 보여줍니다.',
    danceRole: '소무의 아름다움에 반해 파계하는 위선적 승려 역할',
    quote: '불도를 닦은 지 수십 년인데... 아, 저 여인이여!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  somu: {
    name: 'somu',
    koreanName: '소무',
    origin: '봉산탈춤',
    description: '젊은 무희를 상징하는 탈로, 아름답고 요염한 표정이 특징입니다.',
    danceRole: '노장을 유혹하여 파계하게 만드는 요염한 무희 역할',
    quote: '스님, 이리 오시어요~',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  yeongam: {
    name: 'yeongam',
    koreanName: '영감',
    origin: '봉산탈춤',
    description: '늙은 남편을 상징하는 탈로, 첩을 얻어 본처를 홀대하는 가부장적 인물입니다.',
    danceRole: '미얄할미와 덜머리집 사이에서 갈등하는 가부장 역할',
    quote: '에잇, 늙은 년은 물러가라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  
  // 고성오광대
  mundungi: {
    name: 'mundungi',
    koreanName: '문둥이',
    origin: '고성오광대',
    description: '문둥병 환자를 상징하는 탈로, 병으로 일그러진 얼굴이 특징입니다. 사회적 약자의 한을 표현합니다.',
    danceRole: '병자의 슬픔과 한을 춤으로 표현하는 역할',
    quote: '손가락이 다 떨어져 나가도 살아야 하는 이 목숨...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  bibi: {
    name: 'bibi',
    koreanName: '비비',
    origin: '고성오광대',
    description: '양반의 첩을 상징하는 탈로, 요염하고 교태로운 표정이 특징입니다.',
    danceRole: '양반과 말뚝이 사이 갈등의 원인이 되는 첩 역할',
    quote: '나으리, 저만 보시어요~',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  joje: {
    name: 'joje',
    koreanName: '조제',
    origin: '고성오광대',
    description: '제비를 상징하는 탈로, 흥부놀부전의 제비를 연상시키는 독특한 형태입니다.',
    danceRole: '제비의 날렵한 움직임을 춤으로 표현하는 역할',
    quote: '흥부야, 이 박씨를 심어라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  
  // 특수 탈
  cheoyong: {
    name: 'cheoyong',
    koreanName: '처용',
    origin: '처용무',
    description: '신라 헌강왕 때의 인물 처용을 형상화한 탈입니다. 역신을 물리치는 벽사의 의미를 가지며, 궁중무용인 처용무에 사용됩니다.',
    danceRole: '오방색 의상을 입고 역신을 물리치는 궁중무용 역할',
    quote: '역신이여 물러가라! 내 아내를 범한 죄 용서하리니...',
    wikiUrl: 'https://ko.wikipedia.org/wiki/처용무'
  },
  bangsangsi: {
    name: 'bangsangsi',
    koreanName: '방상씨',
    origin: '나례',
    description: '귀신을 쫓는 벽사 의식에 사용되는 탈입니다. 네 개의 황금빛 눈이 특징이며, 곰가죽 옷을 입고 창과 방패를 들고 귀신을 쫓습니다.',
    danceRole: '궁중 나례 의식에서 역귀를 쫓는 벽사의 신 역할',
    quote: '악귀야 물러가라! 사방을 지키노라!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/방상시'
  },
  
  // 기타 유명 탈
  songpa: {
    name: 'songpa',
    koreanName: '송파산대놀이탈',
    origin: '송파산대놀이',
    description: '서울 송파 지역의 산대놀이에 사용되는 탈입니다. 서민적이고 해학적인 표정이 특징입니다.',
    danceRole: '양반 풍자와 파계승 이야기를 펼치는 서울 지역 탈놀이 역할',
    quote: '양반인들 뭐가 다르단 말이오?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/송파산대놀이'
  },
  gangnyeong: {
    name: 'gangnyeong',
    koreanName: '강령탈',
    origin: '강령탈춤',
    description: '황해도 강령 지역의 탈춤에 사용되는 탈입니다. 봉산탈춤과 비슷하지만 더 투박하고 거친 느낌이 특징입니다.',
    danceRole: '역동적인 춤사위로 황해도를 대표하는 탈놀이 역할',
    quote: '에라, 모르겠다! 한바탕 놀아보세!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/강령탈춤'
  },
  tongyeong: {
    name: 'tongyeong',
    koreanName: '통영오광대탈',
    origin: '통영오광대',
    description: '경남 통영 지역의 오광대놀이에 사용되는 탈입니다. 남해안 특유의 개방적이고 활달한 분위기가 느껴집니다.',
    danceRole: '바다 인접 지역 특성의 활기차고 역동적인 탈놀이 역할',
    quote: '바다 사나이가 왔소! 한판 붙어볼까?',
    wikiUrl: 'https://ko.wikipedia.org/wiki/통영오광대'
  },
  suyeong: {
    name: 'suyeong',
    koreanName: '수영야류탈',
    origin: '수영야류',
    description: '부산 수영 지역의 야류(들놀이)에 사용되는 탈입니다. 영남 지역 특유의 강인한 느낌이 특징입니다.',
    danceRole: '정월대보름 마을 축제를 이끄는 부산 대표 민속놀이 역할',
    quote: '영남 사나이 기개를 보여주마!',
    wikiUrl: 'https://ko.wikipedia.org/wiki/수영야류'
  }
};

// 탈 이름으로 정보 찾기 (파일명에서 추출된 이름 지원)
export function findMaskInfo(maskName: string): MaskDetailInfo | null {
  // 숫자 제거 및 정규화 (예: 양반1 -> 양반, Yangban2 -> yangban)
  const cleanName = maskName.replace(/\d+/g, '').trim();
  const normalizedName = cleanName.toLowerCase().replace(/[^a-z가-힣]/g, '');
  
  // 정확한 key 매칭 (영문)
  if (MASK_INFO[normalizedName]) {
    return MASK_INFO[normalizedName];
  }
  
  // 한글 이름으로 정확한 매칭
  for (const [key, info] of Object.entries(MASK_INFO)) {
    if (info.koreanName === cleanName || info.koreanName === normalizedName) {
      return info;
    }
  }
  
  // 부분 매칭 (탈 이름이 포함된 경우)
  for (const [key, info] of Object.entries(MASK_INFO)) {
    const koreanNameNormalized = info.koreanName.toLowerCase();
    if (normalizedName.includes(key) || key.includes(normalizedName) ||
        normalizedName.includes(koreanNameNormalized) || koreanNameNormalized.includes(normalizedName)) {
      return info;
    }
  }
  
  return null;
}
